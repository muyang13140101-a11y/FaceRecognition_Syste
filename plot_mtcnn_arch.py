import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 确保目录存在
os.makedirs("./eval_test", exist_ok=True)

# 基础设置
plt.rcParams['font.size'] = 9.5
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

def draw_terminator(ax, x, y, w, h, text, fc='#F4CCCC', ec='#CC0000'):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.4",
                                 linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontweight='bold', zorder=3)
    return (x, y+h/2), (x+w, y+h/2), (x+w/2, y), (x+w/2, y+h)

def draw_data(ax, x, y, w, h, text, fc='#D9EAD3', ec='#38761D'):
    s = w * 0.12
    pts = [[x+s, y], [x+w, y], [x+w-s, y+h], [x, y+h]]
    poly = patches.Polygon(pts, linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(poly)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontweight='bold', zorder=3)
    return (x+s/2, y+h/2), (x+w-s/2, y+h/2), (x+w/2, y), (x+w/2, y+h)

def draw_process(ax, x, y, w, h, text, fc='#CFE2F3', ec='#2B78E4'):
    rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontweight='bold', zorder=3)
    return (x, y+h/2), (x+w, y+h/2), (x+w/2, y), (x+w/2, y+h)

def draw_decision(ax, x, y, w, h, text, fc='#FFF2CC', ec='#D9A11B'):
    pts = [[x, y+h/2], [x+w/2, y+h], [x+w, y+h/2], [x+w/2, y]]
    poly = patches.Polygon(pts, linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(poly)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontweight='bold', zorder=3)
    return (x, y+h/2), (x+w, y+h/2), (x+w/2, y), (x+w/2, y+h)

def draw_arrow(ax, posA, posB, text="", text_offset=(0, 0.2), color="#444444", txt_color="red"):
    ax.annotate("", xy=posB, xytext=posA,
                arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.5",
                                color=color, lw=1.5), zorder=1)
    if text:
        mx = (posA[0] + posB[0]) / 2
        my = (posA[1] + posB[1]) / 2
        ax.text(mx + text_offset[0], my + text_offset[1], text, ha='center', va='center', 
                fontsize=8.5, color=txt_color, fontweight='bold', zorder=4)

# 使用稍微缩小的画布比例，确保不被边缘裁剪
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 10.0)
ax.axis('off')

# 图形尺寸
w, h = 1.7, 1.0
c1, c2, c3, c4, c5, c6 = 0.5, 2.8, 5.1, 7.4, 9.7, 12.0

# 绘制流程
y1 = 8.0
b1 = draw_terminator(ax, c1, y1, w, h, "开始")
b2 = draw_data(ax, c2, y1, w, h, "输入图像")
b3 = draw_process(ax, c3, y1, w, h, "图像金字塔")
b4 = draw_process(ax, c4, y1, w, h, "P-Net推理")
b5 = draw_decision(ax, c5, y1, w, h, "NMS判定")

draw_arrow(ax, b1[1], b2[0])
draw_arrow(ax, b2[1], b3[0])
draw_arrow(ax, b3[1], b4[0])
draw_arrow(ax, b4[1], b5[0])
draw_arrow(ax, b5[1], (b5[1][0]+0.8, b5[1][1]), "丢弃", text_offset=(0, 0.15))

y2 = 4.5
draw_arrow(ax, b5[2], (b5[2][0], y2+h), "输出候选框", text_offset=(-1.1, 0))

b6 = draw_process(ax, c5, y2, w, h, "Resize 24x24")
b7 = draw_process(ax, c4, y2, w, h, "R-Net精滤")
b8 = draw_decision(ax, c3, y2, w, h, "阈值>0.95?")

draw_arrow(ax, b6[0], b7[1])
draw_arrow(ax, b7[0], b8[1])
draw_arrow(ax, b8[0], (b8[0][0]-0.8, b8[0][1]), "丢弃", text_offset=(0, 0.15))

y3 = 1.2
draw_arrow(ax, b8[2], (b8[2][0], y3+h), "输出框", text_offset=(1.1, 0))

b9 = draw_process(ax, c3, y3, w, h, "Resize 48x48")
b10 = draw_process(ax, c4, y3, w, h, "O-Net推理")
b11_top = draw_data(ax, c5, y3+0.6, w, h*0.7, "边界框(BBox)")
b11_bot = draw_data(ax, c5, y3-0.5, w, h*0.7, "关键点(Points)")
b12 = draw_terminator(ax, c6, y3, w, h, "结束")

draw_arrow(ax, b9[1], b10[0])
draw_arrow(ax, b10[1], b11_top[0])
draw_arrow(ax, b10[1], b11_bot[0])
draw_arrow(ax, b11_top[1], (b12[0][0], b11_top[1][1]))
draw_arrow(ax, b11_bot[1], (b12[0][0], b11_bot[1][1]))

plt.title("图 3-1：MTCNN 级联感知与判别标准流程图", fontsize=12, fontweight='bold', y=0.96)

# 保存为 300 DPI 的高清晰 PNG
save_path = "./eval_test/Fig3-1_Standard_Flowchart.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"高清晰 PNG 已生成，请前往查看: {save_path}")