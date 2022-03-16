# Cylinder 2D steady 模型使用说明
  以下命令在aistudio notebook中定义，请结合环境对应修改。
  
1) 安装paddle-gpu develop版本
    
    !python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
    
2）安装依赖项，在PaddleScience目录下（因为使用的是aistudio,采用了持久化目录）
    %cd PaddleScience_2D_Cylinder_Steady/
    
    !pip install -r requirements.txt
    
3）设置python搜索路径，添加环境变量

   如果是aistudio如下，也可以export或者修改bashrc等。
   
   %env PYTHONPATH=/home/aistudio/PaddleScience
   
   
   
4）执行仿真模型
   
   !python examples/cylinder2d/cylinder2d_steady1.py
