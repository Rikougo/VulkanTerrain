for %%i in (.\shaders\*) do (
    glslc %%i -o .\shaders\bin\%%~ni.spv
)