# Product copy / 产品文案

Use the clarity principles of ASD-STE100 as writing guidance, not as a claim of certified compliance. The Chinese text follows the same principles; English word-count rules are not applied mechanically to Chinese. This project has not verified every term against the official approved dictionary.

借鉴 ASD-STE100 的清晰技术写作原则，不宣称认证或全面符合规范。中英文都应让玩家知道任务、操作、结果和限制。

## Write for learning / 为学习过程写文案

Start with an invitation to try, then connect what the player observes to its technical name. This is a learning product, not an operator manual. Clarity should remove confusion, not curiosity.

先让玩家愿意试，再把观察到的现象与术语联系起来。清晰不是把标题改成定义，也不是把专业知识全部删掉。

- **Before trying:** keep vivid, useful lesson titles and describe the observable goal in everyday language. A metaphor is welcome when the instructions make the task clear; do not replace it with a stack of technical terms.
- **While trying:** use the real module names and give clear instructions without turning every puzzle into a complete wiring recipe. Do not give modules playful aliases that players must later unlearn.
- **When exploring or revisiting:** use the existing explanations and module cards to connect the concrete action to the technical term, then give its precise meaning and formula. These explanations stay available before and after completion; do not assume the player has already solved the lesson.
- 标题和目标负责上手，模块名保持准确，原理和卡片负责深入。不要逐句并排重复“通俗版／专业版”，也不新增模式、强制弹窗或通关后才开放的术语层。
- Put the action and object first in instructions. Write one step per sentence where a sequence matters. Put conditions before the action.
- State the required result in the goal. Put actions in the instructions and causes in the explanation. Do not hide text or remove mathematical explanations to claim a shorter interface.
- Use the actual button name. For example, **Return to workbench / 返回工作台** keeps the circuit and clears this run's results; it does not start a new attempt.
- For an error, name the object and cause, then give an action the game supports. Preserve actual/expected dimensions, limits and units. Do not guess a cause for an unknown error.
- Distinguish no stock in this level from all copies being placed, and distinguish the module limit from a blocked placement. Module details remain available when adding is unavailable, except during a run.
- Keep necessary conditions and negatives. Scores are not probabilities; token IDs are not feature vectors; concatenation is not addition; FFN output is not the result after a residual connection.
- Identify teaching simulations as simulations. Do not describe engine costs as hardware measurements, or fixed teaching weights as a trained language model.

## Terms

| 中文 | English | Use |
| --- | --- | --- |
| 模块 | module | A placed operation or packaged computation |
| 组件架 | module shelf | The existing shelf control; do not rename it in explanatory text |
| 工作台 | workbench | The editing view; 台面 refers to its placement area |
| 输入端口 / 输出端口 | input port / output port | Square input / round output |
| 关卡出口 | lesson outlet | The final output checked against the target |
| 运行 / 暂停 / 继续 / 停止 | Run / Pause / Resume / Stop | The existing playback controls |
| 返回工作台 | Return to workbench | Keep the circuit; clear this run's results |
| 重新搭建 | Rebuild | Restore the current level's initial circuit |
| 仅清除通关标记 | Clear completion badges only | Keep circuits; cannot be undone |
| 全部重新开始 | Start over in all levels | Restore all initial circuits; has a separate undo action |
| 特征 / token / Token ID | feature / token / token ID | Keep the numerical representation distinct from token selection |
| 分数 / 概率 / 注意力权重 | score / probability / attention weight | Do not use interchangeably |
| 首词时间 / 解码速度 | time to first token / decoding speed | Preserve ms and token/s |

## Maintain both languages

Chinese source strings are lookup keys. Update the caller and `locales/en.js` together. For inline `L(zh,en)`, change both arguments. `structure-map.js` has a private translation map and must be checked separately. Use complete parameterized messages rather than concatenating translated sentence fragments. Preserve placeholders and escape dynamic text when inserting HTML.

Check the final loaded lesson, not only its first definition: later curriculum files and `lesson-io.js` can replace titles, boundaries and descriptions. `menu.summary` overrides `summary`; assigning `summary: description` copies the current value. Module-card overrides must be checked in their lesson context. Unused `hints` are not evidence of a visible improvement.

## Examples

The first two rows compare an unpublished, overly technical draft with the final learning copy; they are not changes from the live site. The remaining rows compare the original UI wording with the revised wording.

前两行是未发布的技术化草稿与最终学习文案的对照，不是线上版本的变更；后几行是原有操作文案与改后文案的对照。

| Before | After | Reason |
| --- | --- | --- |
| SiLU 首屏：输出投影结果经 SiLU 计算后的 4 维向量 | 目标：改变数值，保留四个特征。卡片：逐项激活、非线性变换与 SiLU 公式 | Start with the observable result; connect it to the technical term in the existing explanation |
| 残差标题：把 FFN 输出加回输入 | 标题：留一条回来的路。原理：把输出加回输入叫残差连接 | Keep the invitation to explore without losing the precise definition |
| 试跑我的网络 / Run my network | 运行 / Run | One action name in buttons and instructions |
| 重玩本关 / Replay level | 返回工作台 / Return to workbench | Describes the existing action without implying a reset |
| 点连线删除 | 选中连线后点击“删除连线” | States the two required actions |
| 可用数量 0 / 0 · 双击查看说明 | 可用数量：0 / 0。本关不提供此模块。双击或按 Enter 查看说明。 | Gives the actual restriction and preserves the details action |

## Verification

Run `npm test`, including the bilingual UI checks. Review effective copy for every route and optional module, plus errors, tooltips and accessible labels. Check formulas, dimensions, thresholds, inputs, reference answers, inventories and saved-state behavior separately from prose. Update precise text expectations when text changes; do not weaken them to non-empty checks. Inspect the real browser at desktop widths in both languages: mock rendering and an empty missing-translation set do not prove visual or linguistic quality.
