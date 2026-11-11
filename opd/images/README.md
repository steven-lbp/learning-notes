# 图片来源与保存约定

本目录存放 OPD 笔记引用的图片，正文统一使用相对路径，例如：

```markdown
![OPD 流程](images/opd-training-loop.svg)
```

- 流程图优先保留 SVG；截图和生成位图保留 PNG、WebP、JPEG 等实际文件。
- 使用有意义的文件名，避免正文依赖聊天临时链接或本机绝对路径。
- 移动主题时，一起移动 Markdown 和图片目录。

| 文件 | 来源 |
| --- | --- |
| `opd-training-loop.svg` | 原学习对话中模型绘制的 SVG，补充白底、字体与标题 |
| `opsd-training-loop.svg` | 原学习对话中模型绘制的 SVG，补充白底、字体与标题 |
| `teacher-organization.svg` | 根据原对话文字和箭头示意补画的完整关系图 |
| `mimo-deepseek-token-distributions.svg` | 根据原学习对话的 A/B/C 概率示例补画，比较 sampled-token 与 full-vocabulary 蒸馏信号 |
| `source-model-comparison.png` | 原对话中用户上传的文章长截图，保留原文件 |
