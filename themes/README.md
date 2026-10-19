# Coloring Book Themes

This directory contains theme definitions for the coloring book generator.

## Structure

### `themes.json`

Main theme configuration file. Each theme has:

```json
{
  "theme_key": {
    "name": "Display Name",
    "prompts": [
      "prompt 1 description...",
      "prompt 2 description...",
      ...
    ]
  }
}
```

**Fields**:
- `theme_key`: Internal identifier (lowercase, no spaces)
- `name`: Human-readable theme name
- `prompts`: Array of image generation prompts

## Available Themes

Currently includes 17 professional themes:

| Theme | Name | Prompts | Description |
|-------|------|---------|-------------|
| `mandalas` | Mystical Mandalas | 5 | Sacred geometric patterns |
| `animals` | Enchanted Animals | 8 | Ornate animal portraits |
| `nature` | Botanical Gardens | 6 | Flowers, plants, gardens |
| `geometric` | Sacred Geometry | 5 | Complex geometric patterns |
| `fantasy` | Fantasy Realms | 6 | Dragons, unicorns, fairies |
| `patterns` | Relaxing Patterns | 5 | Zentangle, paisley, abstract |
| `inspirational` | Mindful Words | 5 | Motivational words with art |
| `ocean` | Ocean Wonders | 8 | Sea creatures and underwater scenes |
| `flowers` | Blooming Gardens | 8 | Detailed floral arrangements |
| `zen` | Zen & Meditation | 8 | Buddha, meditation, spiritual |
| `christmas` | Christmas Magic | 8 | Holiday and winter themes |
| `halloween` | Halloween Spooky | 8 | Spooky and autumn themes |
| `celtic` | Celtic Knots | 8 | Traditional Celtic knotwork |
| `japanese` | Japanese Art | 8 | Koi, cherry blossoms, pagodas |
| `space` | Cosmic Dreams | 8 | Planets, stars, astronauts |
| `food` | Delicious Treats | 8 | Desserts and sweets |
| `architecture` | Beautiful Buildings | 8 | Iconic buildings and structures |

## Adding Custom Themes

### Method 1: Edit themes.json

Add a new theme to `themes.json`:

```json
{
  "mytheme": {
    "name": "My Custom Theme",
    "prompts": [
      "detailed prompt 1, adult coloring book style, black line art",
      "detailed prompt 2, intricate patterns, clean outlines",
      "prompt 3...",
      "prompt 4...",
      "prompt 5..."
    ]
  }
}
```

**Prompt Guidelines**:
- Include "adult coloring book" or "coloring page"
- Specify "black line art" or "black outlines"
- Mention "intricate", "detailed", "ornate" for complexity
- Add "on white background" or "white background"
- Avoid color names (images should be black & white)
- 5-8 prompts per theme recommended

### Method 2: Create custom_themes.json

Create `themes/custom_themes.json` for your personal themes:

```json
{
  "cats": {
    "name": "Fancy Cats",
    "prompts": [
      "ornate cat with decorative fur patterns..."
    ]
  }
}
```

Then load with:
```python
from pathlib import Path
from coloring_book_generator import load_themes

custom_themes = load_themes(Path("themes/custom_themes.json"))
```

## Theme Best Practices

**DO**:
- Use descriptive, detailed prompts
- Mention "coloring book" or "coloring page" style
- Specify line art characteristics (thick, thin, clean, intricate)
- Include artistic style keywords (zentangle, mandala, ornate)
- Create 5-8 varied prompts per theme
- Test prompts to ensure good line art output

**DON'T**:
- Use color names (red, blue, etc.) - should be black & white
- Make prompts too short or vague
- Include shading/gradient keywords
- Use photo-realistic prompts (won't convert well to line art)
- Create offensive or controversial content

## Example Prompt Anatomy

```
"majestic lion portrait with decorative mane made of intricate patterns and flowers, adult coloring book style, black line art on white"
     ├─ Subject ─┤          ├─────────── Details ──────────┤  ├──── Style ────┤  ├──── Format ────┤
```

**Components**:
1. **Subject**: What to draw (lion portrait)
2. **Details**: Decorative elements (patterns, flowers in mane)
3. **Style**: Art style (adult coloring book)
4. **Format**: Technical specs (black line art on white)

## Validation

After editing `themes.json`, validate with:

```bash
python3 -c "import json; json.load(open('themes/themes.json')); print('Valid JSON')"
```

Or test loading:

```bash
python3 -c "from coloring_book_generator import THEMES; print(f'{len(THEMES)} themes loaded')"
```

## File Format

- **Format**: JSON
- **Encoding**: UTF-8
- **Indent**: 2 spaces
- **Line Endings**: LF (Unix style)

## Version History

- **v1.0**: Initial 7 themes (mandalas, animals, nature, geometric, fantasy, patterns, inspirational)
- **v2.0**: Added 10 new themes (ocean, flowers, zen, christmas, halloween, celtic, japanese, space, food, architecture)

## Validating themes

Beyond the JSON one-liners in [Validation](#validation) above, `validate_themes.py`
(repo root, standard library only) checks a themes file against what the generator
actually reads (`name` and `prompts`) and prints a per-theme report:

```bash
python3 validate_themes.py                           # the bundled themes/themes.json
python3 validate_themes.py themes/custom_themes.json
python3 validate_themes.py --strict                  # warnings fail too
```

Exit status is `0` when clean, `1` on any error (or any warning with `--strict`) and
`2` for a bad command line. Each finding ends with a check id such as `[duplicate-prompt]`.

**Errors** (the generator would fail or silently misbehave): a missing or unreadable
file, invalid UTF-8 or JSON (including a BOM), a top level that is not an object or has
no themes, repeated keys (`json.load` silently keeps the last), theme keys that are not
lowercase slugs, themes that are not objects, missing or wrongly typed `name`/`prompts`,
a blank name or prompt, an empty `prompts` list, a `/` or `\` in `name` (it becomes the
output directory name), a prompt over 1000 characters or a name over 100, and a prompt
repeated within one theme.

**Warnings** (quality heuristics): unknown fields, leading/trailing or repeated
whitespace, fewer than 5 prompts, prompts under 40 or over 330 characters (shipped
prompts are 81-165), names over 40 characters, a prompt ending in punctuation (the
generator appends `, with ...` variations), a prompt or name shared between themes, and
wording that fights black-and-white line art (color names, shading/gradient/gray,
photo-realistic, black backgrounds). Negated phrases such as "no shading" are fine.

Run the validator's own tests with `python3 test_validate_themes.py`.

---

*For more information, see the main project README.md*
