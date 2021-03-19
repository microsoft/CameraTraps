from cx_Freeze import setup, Executable 
  
setup(name = "WII XMP Integration" , 
      version = "3.0" , 
      description = "XMP metadata Writer for WII Images" , 
      executables = [Executable("xmp_integration.py")])