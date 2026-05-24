@echo off
echo ============================================
echo   全格式自进化知识库智能体 - 启动脚本
echo ============================================
echo.

REM 检查虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo [错误] 虚拟环境未找到，正在创建...
    python -m venv venv
)

echo [1/4] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [2/4] 检查后端依赖...
pip install -r backend/requirements.txt -q

echo [3/4] 检查前端依赖...
cd frontend
if not exist "node_modules" (
    echo 正在安装前端依赖（首次运行可能需要几分钟）...
    call npm install
)
cd ..

echo [4/4] 启动服务...
echo.
echo 后端 API 服务: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo 前端界面: http://localhost:3000
echo.
echo 按 Ctrl+C 停止所有服务
echo ============================================
echo.

start "知识库后端" cmd /c "venv\Scripts\python.exe backend\main.py"
cd frontend
start "知识库前端" cmd /c "npx next dev -p 3000"
cd ..

echo 服务已启动！请打开浏览器访问 http://localhost:3000