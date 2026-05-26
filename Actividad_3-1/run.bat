@echo off

REM Ubicarse en la carpeta donde está este archivo .bat
cd /d "%~dp0"

REM Activar ambiente virtual
REM call "C:\Users\cegar\OneDrive\Documentos\Python\MNA\4. Navegación Autónoma\.venv\Scripts\activate.bat"
REM También se puede usar esto si solo está ubicado dos niveles arriba:
call "..\..\.venv\Scripts\activate.bat"

REM Configuración temporal de Webots
set "WEBOTS_HOME=C:\Program Files\Webots"
set "LD_LIBRARY_PATH=C:\Program Files\Webots\lib\controller"
set "PATH=%PATH%;C:\Program Files\Webots\msys64\mingw64\bin"

REM Ejecutar controlador
webots-controller.exe Act_3_1_Equipo2.py --stdout-redirect

pause
