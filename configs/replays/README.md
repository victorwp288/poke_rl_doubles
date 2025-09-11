Place curated replay identifiers here.

- ids.txt: one replay id per line, e.g. `gen9doublesou-12345678`
- urls.txt: one full replay URL per line, e.g. `https://replay.pokemonshowdown.com/gen9doublesou-12345678`

Both files are optional; the fetch script treats missing/empty files as empty.

### Where to find replays (Gen 9 Doubles OU)
1. Open the Showdown Replay Viewer: [replay.pokemonshowdown.com](https://replay.pokemonshowdown.com).
2. In the "Format" field, type `gen9doublesou` (Gen 9 Doubles OU).
3. Search by username or keywords, or open any replay page you want.
4. Copy either:
   - The replay ID, e.g. `gen9doublesou-2032363987`, or
   - The full replay URL, e.g. `https://replay.pokemonshowdown.com/gen9doublesou-2032363987`.

### How to use in this repo
- Paste IDs into `configs/replays/ids.txt` (one per line).
- Or paste URLs into `configs/replays/urls.txt` (one per line).
- Use only Gen 9 Doubles OU replays: IDs should start with `gen9doublesou-`.

### Examples
IDs (`ids.txt`):
```
gen9doublesou-2032231707
gen9doublesou-2032363987
```

URLs (`urls.txt`):
```
https://replay.pokemonshowdown.com/gen9doublesou-2032231707
https://replay.pokemonshowdown.com/gen9doublesou-2032363987
```