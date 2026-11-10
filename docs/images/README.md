# Repository images

These assets present Neural Atelier on the repository homepage.

- `ffn-workbench.jpg`: cropped in-game capture of a connected FFN workbench.
- `matrix-projection.jpg`: cropped in-game capture of the projection puzzle.
- `module-card.jpg`: cropped in-game capture of the gate projection card.
- `social-preview.png`: 1280 × 640 cover composed from the FFN capture and the game's N / Neural Atelier branding.

The captures come from an earlier Chinese-language build. They are not captures of the current English interface. The cover adds typography around a gameplay capture; it does not depict a separate game screen.

To use the cover for GitHub link previews, upload `social-preview.png` in the repository's **Settings → General → Social preview**. Committing it to the repository does not configure GitHub's social preview automatically. The small owner avatar on GitHub is an account-level image, separate from this cover.

## Visual declutter comparison

`visual-declutter/` contains six unedited PNG browser captures taken on 2026-11-10 for the visual-declutter pull request. They are review evidence, not replacement homepage artwork.

- **Before:** clean upstream revision `a049375af9ea7ecad4758d35a66cdd40a9bb7f8f`.
- **After:** the visual, navigation and focus changes committed alongside these images.
- **Environment:** Orca embedded desktop browser on macOS, Simplified Chinese, 1440 × 900 CSS pixels, scale factor 1. Before and After use separate local origins and fresh per-level progress for these scenes.
- Each PNG preserves the entire viewport at its original size: no cropping, annotations or compositing. Scrollable menu and sidebar content outside the viewport is not stitched into the image. The game keeps its default rendering and animation behavior; moving decoration may be at a different animation frame.

| Pair | Matching scene and interaction |
| --- | --- |
| `home-before.png` / `home-after.png` | Home screen with `wire` selected. Tab once, then reload, reproduces the initial-focus outline before the fix; after the fix, the heading receives initial focus and Tab still focuses the entry button visibly. |
| `menu-before.png` / `menu-after.png` | Enter from Home; Main track, first learning stage, `wire` current, no completed lessons. The After image shows the denser cards and collapsed group explanation. |
| `encoder-before.png` / `encoder-after.png` | `encoder-layer`, no placed modules or connections, default camera/zoom and shelf size. The After image shows the wider shelf, shorter always-visible instructions, collapsed theory and chapter return button. |
