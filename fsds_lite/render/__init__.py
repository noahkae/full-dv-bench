"""
3D rendering with moderngl (OpenGL 3.3 core). Optional: everything else in
fsds_lite works without it.

    meshes.py         procedural geometry (cones, car, ground) - numpy only
    renderer.py       shaders + draw calls; works with any moderngl context
    camera_sensor.py  offscreen camera images on a background thread
    viewer3d.py       interactive 3D view (QOpenGLWidget)
"""
