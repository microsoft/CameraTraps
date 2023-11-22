from cx_Freeze import setup, Executable 
  
setup(name = "XMP Integration" , 
      version = "3.0" , 
      description = "XMP metadata writer" , 
      executables = [Executable("xmp_integration.py")])