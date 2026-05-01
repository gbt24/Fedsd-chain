# FedTracker WebUI 设计说明 📋

## 🎨 设计理念

FedTracker WebUI 采用豆包风格的现代化设计，核心理念为：

### 1. 清新简约

- **色彩选择**：采用紫粉渐变作为主色调，营造柔和现代的科技感
- **圆角设计**：所有元素使用 20px 大圆角，避免尖锐生硬的视觉效果
- **适当留白**：界面元素间保持舒适间距，避免拥挤感

### 2. 流畅动态

- **渐变动画**：背景采用流动渐变，15秒循环动画
- **悬停效果**：所有按钮和卡片配以平滑的悬停变换
- **过渡效果**：使用 cubic-bezier 缓动函数，营造自然流畅感

### 3. 友好交互

- **Emoji 图标**：使用表情符号增加亲和力
- **状态提示**：清晰的成功/失败状态反馈
- **分步引导**：通过折叠面板减少视觉负担，按需展开

## 🎨 配色方案

### 主色调

```css
/* 渐变主色 */
--primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* 辅助渐变 */
--secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);

/* 卡片背景 */
--card-bg: rgba(255, 255, 255, 0.95);

/* 文字颜色 */
--text-primary: #1a1a2e;    /* 主要文字 */
--text-secondary: #4a4a6a;   /* 次要文字 */
```

### 功能色

```css
/* 成功状态 */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* 错误状态 */
background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);

/* 信息提示 */
background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
```

## 📐 布局结构

### 整体架构

```
┌─────────────────────────────────────────┐
│          Header (Logo + Title)          │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────┬──────────┬──────────┐     │
│  │Tab 1:   │Tab 2:    │Tab 3:    │     │
│  │ 图片生成│ 泄漏模拟 │ 所有者识别│     │
│  └─────────┴──────────┴──────────┘     │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │      Content Area               │   │
│  │   (Controls + Outputs)          │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
         Footer (Version + Links)
```

### 三栏布局

```
┌────────────────┬─────────────────────────┐
│   控制面板      │      结果展示区          │
│   (Controls)   │      (Outputs)          │
│                │                         │
│   - 模型选择    │   - 图片画廊             │
│   - 参数配置    │   - 状态消息             │
│   - 高级选项    │   - 详细信息             │
│                │                         │
│   Ratio: 2     │   Ratio: 3              │
└────────────────┴─────────────────────────┘
```

## 🧩 组件设计

### 1. 卡片容器

- 背景：半透明白色（95%不透明度）+ 模糊滤镜
- 阴影：柔和的紫色调阴影
- 边框：1px 白色半透明边框
- 圆角：20px

### 2. 输入框

- 圆角：12px
- 边框：2px 半透明紫色
- 聚焦：紫色边框 + 光晕效果
- 背景色：纯净白色

### 3. 按钮

#### 主要按钮

```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
border-radius: 12px;
padding: 14px 32px;
font-weight: 600;
box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
```

- 悬停：上移 2px + 增强阴影
- 点击：恢复原位

#### 次要按钮

```css
background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
border: 2px solid rgba(102, 126, 234, 0.3);
color: #667eea;
```

### 4. 图片画廊

- 背景渐变：柔和的粉紫渐变
- 图片圆角：12px
- 悬停效果：放大 2% + 阴影增强

### 5. 状态消息

#### 成功状态

```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
color: white;
border-radius: 12px;
```

#### 错误状态

```css
background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
color: white;
border-radius: 12px;
```

## 📱 响应式设计

### 桌面端 (>768px)

- 三栏布局完整展示
- 图片画廊 2 列显示
- 侧边控制面板可见

### 移动端 (<768px)

- 单栏布局
- 图片画廊自适应列数
- 标签页缩小字体
- 头部标题缩小

```css
@media (max-width: 768px) {
    #header-title { font-size: 28px !important; }
    .tabitem { padding: 20px !important; }
    .tabs > .tab-nav > button {
        padding: 10px 16px !important;
        font-size: 13px !important;
    }
}
```

## ✨ 动画效果

### 1. 背景渐变动画

```css
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* 时长：15秒，循环播放 */
```

### 2. 淡入动画

```css
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
```

### 3. 按钮悬停

```css
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
transform: translateY(-2px);
```

## 🎯 可访问性考虑

### 1. 色彩对比度

- 主文字与背景对比度 >7:1
- 次要文字与背景对比度 >4.5:1

### 2. 交互反馈

- 所有交互元素配以明确的悬停状态
- 错误状态使用红色系提示
- 成功状态使用蓝紫色系提示

### 3. 字体选择

- 优先使用系统字体：PingFang SC / Noto Sans SC
- 回退到系统默认字体
- 标题使用粗体，正文使用常规字重

## 📊 性能优化

### 1. CSS 优化

- 使用 CSS 变量减少重复
- 避免过度使用阴影和模糊（仅在卡片上使用）
- 使用 transform 而非 left/top 进行动画

### 2. 图片优化

- 使用 object-fit: contain 避免图片拉伸
- 懒加载图片内容
- 使用适当尺寸的缩略图

### 3. 动画优化

- 使用 CSS 动画而非 JavaScript 动画
- 使用 will-change 提示浏览器优化
- 避免在动画中使用昂贵的属性（如 box-shadow 多次变化）

## 🔧 扩展指南

### 添加新标签页

```python
with gr.Tab("🆕 新功能"):
    with gr.Row():
        with gr.Column(scale=2):
            # 控制面板
            pass
        with gr.Column(scale=3):
            # 结果展示
            pass
```

### 添加新组件

```python
new_component = gr.Component(
    label="组件名称",
    info="提示信息",
    # 使用自定义样式
    elem_classes=["custom-class"]
)
```

### 自定义样式

在 `CUSTOM_CSS` 中添加：

```css
.custom-class {
    /* 自定义样式 */
}
```

## 📝 最佳实践

1. **保持一致性**：所有类似组件使用相同样式
2. **层级清晰**：使用折叠面板管理复杂内容
3. **即时反馈**：用户操作后提供明确的状态提示
4. **错误友好**：错误信息清晰且提供解决建议
5. **性能优先**：避免不必要的重渲染和计算

---

**设计团队**: FedTracker UI/UX Design Team  
**版本**: 1.0.0  
**最后更新**: 2024