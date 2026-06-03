import graphviz

def generate_strict_flowchart():
    # 1. 初始化画布：严格限定 4:3 横向比例，300 DPI 论文级高清输出
    dot = graphviz.Digraph('MTCNN_Pipeline', format='png')
    dot.attr(size='8,6!', ratio='fill', dpi='300', rankdir='LR') 
    
    # 全局节点默认属性 (默认矩形执行框)
    dot.attr('node', style='filled', fillcolor='#ffffff', color='black', penwidth='1.5')
    dot.attr('edge', color='black', penwidth='1.2', arrowsize='0.8')

    # ==========================================
    # 核心机密：底层字体隔离渲染 (已修复空标签 Bug)
    # 强制英文数字使用 Times New Roman，中文使用 SimSun (宋体)
    # 五号字体在排版学中标准对应 10.5 pt
    # ==========================================
    def format_strict_text(en1, cn1, en2="", cn2=""):
        pt = "10.5"
        en_font = "Times New Roman"
        cn_font = "SimSun"
        
        # 局部闭包函数：拦截空字符串，防止 Graphviz 语法崩溃
        def wrap_font(text, font):
            if not text:  # 如果是空字符串，直接返回空，绝不生成空 <FONT> 标签
                return ""
            return f'<FONT FACE="{font}" POINT-SIZE="{pt}">{text}</FONT>'
        
        line1 = wrap_font(en1, en_font) + wrap_font(cn1, cn_font)
        line2_content = wrap_font(en2, en_font) + wrap_font(cn2, cn_font)
        
        if line2_content:
            line2 = f'<BR/>{line2_content}'
        else:
            line2 = ""
            
        return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD ALIGN="CENTER">{line1}{line2}</TD></TR></TABLE>>'

    # ==========================================
    # 2. 定义标准程序图元节点
    # ==========================================
    
    # 模块起止与输入输出
    dot.node('Start', label=format_strict_text('MTCNN', ' 模块启动'), shape='ellipse', fillcolor='#e1f5fe')
    dot.node('Input', label=format_strict_text('', '输入质量预过滤后的有效视频帧'), shape='parallelogram', fillcolor='#fff3e0')
    dot.node('Pyr', label=format_strict_text('Image Pyramid', ' 图像金字塔构建'), shape='box')
    
    # P-Net 阶段
    dot.node('PNet', label=format_strict_text('P-Net', '：轻量级全卷积滑窗初步扫描'), shape='box')
    dot.node('PDec', label=format_strict_text('P-Net ', '置信度判定'), shape='diamond', fillcolor='#e8f5e9')
    dot.node('PNMS', label=format_strict_text('NMS', ' 筛选与边界框初步回归'), shape='box')
    
    # R-Net 阶段
    dot.node('Crop24', label=format_strict_text('', '裁剪候选区域并缩放至 ', '24x24', ' 像素'), shape='box')
    dot.node('RNet', label=format_strict_text('R-Net', '：全连接层空间细化判别'), shape='box')
    dot.node('RDec', label=format_strict_text('R-Net ', '置信度判定'), shape='diamond', fillcolor='#e8f5e9')
    dot.node('RNMS', label=format_strict_text('NMS', ' 筛选与边界框二次微调'), shape='box')
    
    # O-Net 阶段
    dot.node('Crop48', label=format_strict_text('', '裁剪候选区域并缩放至 ', '48x48', ' 像素'), shape='box')
    dot.node('ONet', label=format_strict_text('O-Net', '：最终判别与面部特征回归'), shape='box')
    dot.node('ODec', label=format_strict_text('O-Net ', '置信度判定'), shape='diamond', fillcolor='#e8f5e9')
    dot.node('ONMS', label=format_strict_text('NMS', ' 提取最优框与五点坐标'), shape='box')
    
    # 结束与抛弃节点
    dot.node('Drop', label=format_strict_text('', '丢弃负样本背景框'), shape='ellipse', style='dashed', fillcolor='#eeeeee')
    dot.node('Output', label=format_strict_text('BBox', ' 与 5 点拓扑坐标输出'), shape='parallelogram', fillcolor='#fff3e0')
    dot.node('End', label=format_strict_text('', '送入 5 点相似变换对齐模块'), shape='ellipse', fillcolor='#e1f5fe')
    
    # ==========================================
    # 3. 构建逻辑流向 (Edges)
    # ==========================================
    dot.edge('Start', 'Input')
    dot.edge('Input', 'Pyr')
    dot.edge('Pyr', 'PNet')
    dot.edge('PNet', 'PDec')
    
    # Yes 逻辑主线
    dot.edge('PDec', 'PNMS', label=format_strict_text('Y', ''))
    dot.edge('PNMS', 'Crop24')
    dot.edge('Crop24', 'RNet')
    dot.edge('RNet', 'RDec')
    
    dot.edge('RDec', 'RNMS', label=format_strict_text('Y', ''))
    dot.edge('RNMS', 'Crop48')
    dot.edge('Crop48', 'ONet')
    dot.edge('ONet', 'ODec')
    
    dot.edge('ODec', 'ONMS', label=format_strict_text('Y', ''))
    dot.edge('ONMS', 'Output')
    dot.edge('Output', 'End')
    
    # No 逻辑分流
    dot.edge('PDec', 'Drop', label=format_strict_text('N', ''))
    dot.edge('RDec', 'Drop', label=format_strict_text('N', ''))
    dot.edge('ODec', 'Drop', label=format_strict_text('N', ''))

    # 4. 执行渲染
    output_filename = 'Fig3_MTCNN_Pipeline_Strict'
    dot.render(output_filename, cleanup=True)
    print(f"✅ 生成成功！严格版流程图已保存为 {output_filename}.png")

if __name__ == '__main__':
    generate_strict_flowchart()