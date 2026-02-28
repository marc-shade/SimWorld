"""SceneKit Metal offscreen renderer for SimWorld macOS backend."""
import ctypes
import io
import math

import numpy as np
from PIL import Image

try:
    import SceneKit
    import Quartz
    import Metal
    from Foundation import NSData
    from AppKit import NSBitmapImageRep, NSColor
    SCENEKIT_AVAILABLE = True
except ImportError:
    SCENEKIT_AVAILABLE = False


class SceneKitRenderer:
    """Offscreen 3D renderer using Apple SceneKit + Metal.

    Uses SCNRenderer for headless GPU-accelerated rendering without opening
    any windows. Falls back to software rasterization when PyObjC/SceneKit
    is unavailable.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self._nodes: dict[str, object] = {}
        self._object_colors: dict[str, tuple] = {}
        self._next_mask_color = 1

        if not SCENEKIT_AVAILABLE:
            self._scene = None
            self._renderer = None
            self._device = None
            self._camera_nodes: dict[int, object] = {}
            return

        # Metal device
        self._device = Metal.MTLCreateSystemDefaultDevice()

        # Scene
        self._scene = SceneKit.SCNScene.scene()

        # Offscreen renderer bound to Metal
        self._renderer = SceneKit.SCNRenderer.rendererWithDevice_options_(
            self._device, None
        )
        self._renderer.setScene_(self._scene)
        self._renderer.setAutoenablesDefaultLighting_(True)

        # Lighting, ground, sky, camera
        self._setup_lighting()
        self._setup_ground()
        self._setup_sky()
        self._camera_nodes: dict[int, object] = {}
        self._setup_default_camera()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_lighting(self):
        """Add directional sunlight and ambient fill."""
        # Directional (sun)
        sun = SceneKit.SCNLight.light()
        sun.setType_(SceneKit.SCNLightTypeDirectional)
        sun.setIntensity_(1000)
        sun.setCastsShadow_(True)
        sun.setShadowMapSize_(Quartz.CGSizeMake(2048, 2048))

        sun_node = SceneKit.SCNNode.node()
        sun_node.setLight_(sun)
        sun_node.setPosition_(SceneKit.SCNVector3Make(0, 1000, 0))
        sun_node.setEulerAngles_(SceneKit.SCNVector3Make(-0.785, 0.785, 0))
        self._scene.rootNode().addChildNode_(sun_node)

        # Ambient fill
        ambient = SceneKit.SCNLight.light()
        ambient.setType_(SceneKit.SCNLightTypeAmbient)
        ambient.setIntensity_(300)
        ambient_color = NSColor.colorWithRed_green_blue_alpha_(0.8, 0.8, 0.9, 1.0)
        ambient.setColor_(ambient_color)

        ambient_node = SceneKit.SCNNode.node()
        ambient_node.setLight_(ambient)
        self._scene.rootNode().addChildNode_(ambient_node)

    def _setup_ground(self):
        """Add a large horizontal ground plane."""
        plane = SceneKit.SCNPlane.planeWithWidth_height_(20000, 20000)
        mat = SceneKit.SCNMaterial.material()
        ground_color = NSColor.colorWithRed_green_blue_alpha_(0.3, 0.35, 0.3, 1.0)
        mat.diffuse().setContents_(ground_color)
        mat.setDoubleSided_(True)
        plane.setFirstMaterial_(mat)

        ground = SceneKit.SCNNode.nodeWithGeometry_(plane)
        ground.setPosition_(SceneKit.SCNVector3Make(0, -1, 0))
        # Rotate from vertical (XY) to horizontal (XZ)
        ground.setEulerAngles_(SceneKit.SCNVector3Make(-math.pi / 2, 0, 0))
        self._scene.rootNode().addChildNode_(ground)

    def _setup_sky(self):
        """Set a light-blue sky background."""
        sky = NSColor.colorWithRed_green_blue_alpha_(0.6, 0.75, 0.95, 1.0)
        self._scene.background().setContents_(sky)

    def _setup_default_camera(self):
        """Create the default camera (id=0)."""
        camera = SceneKit.SCNCamera.camera()
        camera.setFieldOfView_(90.0)
        camera.setZNear_(1.0)
        camera.setZFar_(50000.0)

        cam_node = SceneKit.SCNNode.node()
        cam_node.setCamera_(camera)
        cam_node.setPosition_(SceneKit.SCNVector3Make(0, 500, 0))
        cam_node.setEulerAngles_(SceneKit.SCNVector3Make(0, 0, 0))

        self._scene.rootNode().addChildNode_(cam_node)
        self._camera_nodes[0] = cam_node

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def _get_or_create_camera(self, cam_id: int):
        """Return camera node, creating on first access."""
        if cam_id in self._camera_nodes:
            return self._camera_nodes[cam_id]

        camera = SceneKit.SCNCamera.camera()
        camera.setFieldOfView_(90.0)
        camera.setZNear_(1.0)
        camera.setZFar_(50000.0)

        cam_node = SceneKit.SCNNode.node()
        cam_node.setCamera_(camera)
        cam_node.setPosition_(SceneKit.SCNVector3Make(0, 500, 0))
        cam_node.setEulerAngles_(SceneKit.SCNVector3Make(0, 0, 0))

        if SCENEKIT_AVAILABLE and self._scene is not None:
            self._scene.rootNode().addChildNode_(cam_node)
        self._camera_nodes[cam_id] = cam_node
        return cam_node

    def update_camera(self, cam_id: int, position: np.ndarray = None,
                      rotation: np.ndarray = None, fov: float = None,
                      resolution: tuple = None):
        """Update camera position/rotation/fov."""
        if not SCENEKIT_AVAILABLE:
            return
        cam_node = self._get_or_create_camera(cam_id)
        if position is not None:
            # SimWorld (x, y, z) with z-up -> SceneKit y-up: (x, z, y)
            cam_node.setPosition_(SceneKit.SCNVector3Make(
                float(position[0]), float(position[2]), float(position[1])
            ))
        if rotation is not None:
            cam_node.setEulerAngles_(SceneKit.SCNVector3Make(
                math.radians(float(rotation[0])),
                math.radians(float(rotation[1])),
                math.radians(float(rotation[2]))
            ))
        if fov is not None and cam_node.camera():
            cam_node.camera().setFieldOfView_(fov)

    # ------------------------------------------------------------------
    # Object mask color allocation
    # ------------------------------------------------------------------

    def _allocate_mask_color(self, name: str) -> tuple:
        """Assign a unique RGB color for segmentation masks."""
        if name in self._object_colors:
            return self._object_colors[name]
        c = self._next_mask_color
        self._next_mask_color += 1
        r = c & 0xFF
        g = (c >> 8) & 0xFF
        b = (c >> 16) & 0xFF
        self._object_colors[name] = (r, g, b)
        return (r, g, b)

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def _make_material(self, r: float, g: float, b: float):
        """Create an SCNMaterial with the given 0-1 RGB diffuse color."""
        mat = SceneKit.SCNMaterial.material()
        color = NSColor.colorWithRed_green_blue_alpha_(r, g, b, 1.0)
        mat.diffuse().setContents_(color)
        return mat

    def add_object(self, name: str, prefab: str, position: np.ndarray,
                   scale: np.ndarray, color: tuple = (200, 200, 200),
                   object_type: str = 'static'):
        """Add a generic object to the scene.

        Maps object archetypes to appropriate SceneKit geometry primitives.
        """
        if not SCENEKIT_AVAILABLE:
            return
        self._allocate_mask_color(name)

        prefab_lower = (prefab or '').lower()
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])

        if object_type == 'humanoid' or 'pedestrian' in prefab_lower or 'user_agent' in prefab_lower:
            geometry = SceneKit.SCNCapsule.capsuleWithCapRadius_height_(25, 170)
        elif object_type == 'vehicle' or 'vehicle' in prefab_lower:
            geometry = SceneKit.SCNBox.boxWithWidth_height_length_chamferRadius_(
                200 * sx, 150 * sy, 450 * sz, 5
            )
        elif object_type == 'scooter' or 'scooter' in prefab_lower:
            geometry = SceneKit.SCNBox.boxWithWidth_height_length_chamferRadius_(
                60 * sx, 100 * sy, 180 * sz, 3
            )
        elif object_type == 'robot' or 'dog' in prefab_lower:
            geometry = SceneKit.SCNBox.boxWithWidth_height_length_chamferRadius_(
                80 * sx, 60 * sy, 120 * sz, 5
            )
        else:
            geometry = SceneKit.SCNBox.boxWithWidth_height_length_chamferRadius_(
                100 * sx, 100 * sy, 100 * sz, 0
            )

        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        geometry.setFirstMaterial_(self._make_material(r, g, b))

        node = SceneKit.SCNNode.nodeWithGeometry_(geometry)
        node.setPosition_(SceneKit.SCNVector3Make(
            float(position[0]), float(position[2]), float(position[1])
        ))

        self._scene.rootNode().addChildNode_(node)
        self._nodes[name] = node

    def update_object(self, name: str, position: np.ndarray = None,
                      rotation: np.ndarray = None, scale: np.ndarray = None,
                      color: tuple = None):
        """Update an existing object's transform or color."""
        if not SCENEKIT_AVAILABLE or name not in self._nodes:
            return
        node = self._nodes[name]

        if position is not None:
            node.setPosition_(SceneKit.SCNVector3Make(
                float(position[0]), float(position[2]), float(position[1])
            ))
        if rotation is not None:
            node.setEulerAngles_(SceneKit.SCNVector3Make(
                math.radians(float(rotation[0])),
                math.radians(float(rotation[1])),
                math.radians(float(rotation[2]))
            ))
        if color is not None and node.geometry() and node.geometry().firstMaterial():
            r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            new_color = NSColor.colorWithRed_green_blue_alpha_(r, g, b, 1.0)
            node.geometry().firstMaterial().diffuse().setContents_(new_color)

    def remove_object(self, name: str):
        """Remove an object from the scene."""
        if name in self._nodes:
            if SCENEKIT_AVAILABLE:
                self._nodes[name].removeFromParentNode()
            del self._nodes[name]
            self._object_colors.pop(name, None)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_camera(self, cam_id: int, viewmode: str = 'lit',
                      width: int = None, height: int = None) -> np.ndarray:
        """Render a camera view to a numpy array.

        Args:
            cam_id: Camera identifier.
            viewmode: 'lit' (full shading), 'depth' (depth buffer),
                      'object_mask' (per-object segmentation).
            width: Override render width.
            height: Override render height.

        Returns:
            RGB uint8 (H, W, 3) for lit/mask, float32 (H, W) for depth.
        """
        w = width or self.width
        h = height or self.height

        if not SCENEKIT_AVAILABLE:
            return self._fallback_render(w, h, viewmode)

        cam_node = self._get_or_create_camera(cam_id)
        self._renderer.setPointOfView_(cam_node)

        size = Quartz.CGSizeMake(w, h)

        if viewmode == 'depth':
            return self._render_depth(size)
        elif viewmode == 'object_mask':
            return self._render_object_mask(w, h)
        else:
            return self._render_lit(size)

    def _render_lit(self, size) -> np.ndarray:
        """Standard lit render with 4x MSAA."""
        snapshot = self._renderer.snapshotAtTime_withSize_antialiasingMode_(
            0, size, SceneKit.SCNAntialiasingModeMultisampling4X
        )
        return self._nsimage_to_numpy(snapshot)

    def _render_depth(self, size) -> np.ndarray:
        """Approximate depth from luminance of a flat render."""
        snapshot = self._renderer.snapshotAtTime_withSize_antialiasingMode_(
            0, size, SceneKit.SCNAntialiasingModeNone
        )
        img = self._nsimage_to_numpy(snapshot)
        depth = np.mean(img.astype(np.float32), axis=2)
        return depth

    def _render_object_mask(self, w: int, h: int) -> np.ndarray:
        """Render per-object segmentation mask using flat unique colors."""
        # Save original materials
        originals: dict[str, object] = {}
        for name, node in self._nodes.items():
            geom = node.geometry()
            if geom is None:
                continue
            first_mat = geom.firstMaterial()
            if first_mat is None:
                continue
            originals[name] = first_mat

            mask_mat = SceneKit.SCNMaterial.material()
            mask_mat.setLightingModelName_(SceneKit.SCNLightingModelConstant)
            cr, cg, cb = self._object_colors.get(name, (0, 0, 0))
            mask_color = NSColor.colorWithRed_green_blue_alpha_(
                cr / 255.0, cg / 255.0, cb / 255.0, 1.0
            )
            mask_mat.diffuse().setContents_(mask_color)
            geom.setFirstMaterial_(mask_mat)

        size = Quartz.CGSizeMake(w, h)
        snapshot = self._renderer.snapshotAtTime_withSize_antialiasingMode_(
            0, size, SceneKit.SCNAntialiasingModeNone
        )
        result = self._nsimage_to_numpy(snapshot)

        # Restore originals
        for name, mat in originals.items():
            if name in self._nodes:
                geom = self._nodes[name].geometry()
                if geom is not None:
                    geom.setFirstMaterial_(mat)

        return result

    def _nsimage_to_numpy(self, nsimage) -> np.ndarray:
        """Convert an NSImage snapshot to an (H, W, 3) uint8 numpy array."""
        tiff_data = nsimage.TIFFRepresentation()
        bitmap = NSBitmapImageRep.imageRepWithData_(tiff_data)

        w = int(bitmap.pixelsWide())
        h = int(bitmap.pixelsHigh())
        samples = int(bitmap.samplesPerPixel())

        raw = bitmap.bitmapData()
        byte_count = w * h * samples

        if isinstance(raw, (bytes, memoryview)):
            arr = np.frombuffer(bytes(raw)[:byte_count], dtype=np.uint8).reshape(h, w, samples)
        else:
            buf = (ctypes.c_uint8 * byte_count).from_address(int(raw))
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, samples)

        if samples >= 4:
            return arr[:, :, :3].copy()
        return arr.copy()

    def _fallback_render(self, w: int, h: int, viewmode: str) -> np.ndarray:
        """Software fallback producing simple colored images."""
        if viewmode == 'depth':
            return np.ones((h, w), dtype=np.float32) * 1000.0

        img = np.full((h, w, 3), fill_value=135, dtype=np.uint8)
        # Simple ground in the lower half
        img[h // 2:, :] = [100, 130, 100]

        # Draw block rectangles for any tracked objects
        rng = np.random.RandomState(42)
        idx = 0
        for name in self._nodes:
            color = rng.randint(80, 220, size=3)
            bx = 50 + (idx * 60) % (w - 100)
            by = h // 4 + (idx * 40) % (h // 2)
            bw, bh = 40, 60
            y0 = max(0, by)
            y1 = min(h, by + bh)
            x0 = max(0, bx)
            x1 = min(w, bx + bw)
            if viewmode == 'object_mask':
                mc = self._object_colors.get(name, (idx + 1, 0, 0))
                img[y0:y1, x0:x1] = mc
            else:
                img[y0:y1, x0:x1] = color
            idx += 1

        return img

    # ------------------------------------------------------------------
    # City-specific helpers (used by scene_loader)
    # ------------------------------------------------------------------

    def add_building(self, name: str, x: float, y: float, width: float,
                     height: float, elevation: float = 0, rotation: float = 0,
                     color: tuple = (180, 180, 190)):
        """Add a building as a vertical box.

        Args:
            name: Unique identifier.
            x: Left edge X (citygen coords).
            y: Top edge Y (citygen coords).
            width: Footprint width.
            height: Footprint depth.
            elevation: Base elevation above ground.
            rotation: Yaw rotation in degrees.
            color: RGB (0-255).
        """
        if not SCENEKIT_AVAILABLE:
            self._nodes[name] = True  # track for fallback
            self._allocate_mask_color(name)
            return
        self._allocate_mask_color(name)

        # Derive building height from footprint
        building_h = max(width, height) * 0.8 + 50

        geometry = SceneKit.SCNBox.boxWithWidth_height_length_chamferRadius_(
            float(width), float(building_h), float(height), 0
        )
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        geometry.setFirstMaterial_(self._make_material(r, g, b))

        node = SceneKit.SCNNode.nodeWithGeometry_(geometry)
        cx = x + width / 2
        cz = y + height / 2
        node.setPosition_(SceneKit.SCNVector3Make(
            float(cx), float(building_h / 2 + elevation), float(cz)
        ))

        if rotation != 0:
            node.setEulerAngles_(SceneKit.SCNVector3Make(
                0, math.radians(rotation), 0
            ))

        self._scene.rootNode().addChildNode_(node)
        self._nodes[name] = node

    def add_road(self, name: str, start_x: float, start_y: float,
                 end_x: float, end_y: float, width: float = 50,
                 is_highway: bool = False):
        """Add a road segment as a flat plane between two points."""
        if not SCENEKIT_AVAILABLE:
            self._nodes[name] = True
            return

        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return

        road_width = width * (2.0 if is_highway else 1.0)
        geometry = SceneKit.SCNPlane.planeWithWidth_height_(
            float(road_width), float(length)
        )

        gray = 0.25 if is_highway else 0.35
        mat = self._make_material(gray, gray, gray)
        mat.setDoubleSided_(True)
        geometry.setFirstMaterial_(mat)

        node = SceneKit.SCNNode.nodeWithGeometry_(geometry)
        mx = (start_x + end_x) / 2
        mz = (start_y + end_y) / 2
        node.setPosition_(SceneKit.SCNVector3Make(float(mx), 0.5, float(mz)))

        angle = math.atan2(dy, dx)
        node.setEulerAngles_(SceneKit.SCNVector3Make(
            -math.pi / 2, 0, float(angle)
        ))

        self._scene.rootNode().addChildNode_(node)
        self._nodes[name] = node

    def add_element(self, name: str, x: float, y: float, width: float,
                    height: float, element_type: str = '',
                    color: tuple = (100, 180, 100)):
        """Add a city element (tree, furniture, obstacle, etc.)."""
        if not SCENEKIT_AVAILABLE:
            self._nodes[name] = True
            self._allocate_mask_color(name)
            return
        self._allocate_mask_color(name)

        etype = element_type.lower()

        if 'tree' in etype:
            # Trees as cones
            canopy_h = 60.0
            geometry = SceneKit.SCNCone.coneWithTopRadius_bottomRadius_height_(
                5, 25, canopy_h
            )
            # Deterministic green from name hash
            seed = hash(name) & 0xFFFF
            green_var = 0.5 + (seed % 100) / 333.0
            mat = self._make_material(0.2, green_var, 0.15)
        elif 'parking' in etype:
            geometry = SceneKit.SCNPlane.planeWithWidth_height_(
                float(width), float(height)
            )
            mat = self._make_material(0.2, 0.2, 0.2)
            mat.setDoubleSided_(True)
        else:
            # Generic small box for furniture, blockers, hydrants, etc.
            size = max(min(width, height), 1.0)
            geometry = SceneKit.SCNBox.boxWithWidth_height_length_chamferRadius_(
                float(size), float(size * 0.5), float(size), 1
            )
            r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            mat = self._make_material(r, g, b)

        geometry.setFirstMaterial_(mat)
        node = SceneKit.SCNNode.nodeWithGeometry_(geometry)

        if 'parking' in etype:
            # Flat on the ground
            node.setPosition_(SceneKit.SCNVector3Make(float(x), 0.2, float(y)))
            node.setEulerAngles_(SceneKit.SCNVector3Make(-math.pi / 2, 0, 0))
        elif 'tree' in etype:
            node.setPosition_(SceneKit.SCNVector3Make(float(x), 30.0, float(y)))
        else:
            node.setPosition_(SceneKit.SCNVector3Make(float(x), 1.0, float(y)))

        self._scene.rootNode().addChildNode_(node)
        self._nodes[name] = node

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_png(image: np.ndarray) -> bytes:
        """Encode numpy image as RGBA PNG bytes.

        The SimWorld client's _decode_png assumes RGBA and strips alpha,
        so we always output 4-channel PNG.
        """
        if image.dtype == np.float32:
            lo, hi = image.min(), image.max()
            span = hi - lo if hi > lo else 1.0
            normalized = ((image - lo) / span * 255).astype(np.uint8)
            pil_img = Image.fromarray(normalized, mode='L')
        elif image.ndim == 3 and image.shape[2] == 3:
            # Add alpha channel (fully opaque) for SimWorld client compatibility
            alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.concatenate([image, alpha], axis=2)
            pil_img = Image.fromarray(rgba, mode='RGBA')
        else:
            pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return buf.getvalue()

    @staticmethod
    def encode_bmp(image: np.ndarray) -> bytes:
        """Encode numpy image as BMP bytes."""
        if image.dtype == np.float32:
            lo, hi = image.min(), image.max()
            span = hi - lo if hi > lo else 1.0
            normalized = ((image - lo) / span * 255).astype(np.uint8)
            pil_img = Image.fromarray(normalized, mode='L')
        else:
            pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format='BMP')
        return buf.getvalue()

    @staticmethod
    def encode_npy(array: np.ndarray) -> bytes:
        """Encode numpy array in .npy format."""
        buf = io.BytesIO()
        np.save(buf, array)
        return buf.getvalue()
