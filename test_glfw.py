import glfw

if glfw.init():
    print("GLFW is installed and can be initialized.")
    glfw.terminate()
else:
    print("Failed to initialize GLFW.")