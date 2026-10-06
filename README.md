# Ollie the Owl

A 3D northern saw-whet owl mascot for agent chat UIs. Ollie sits in the chat and doubles as the loading indicator: it flies laps while the agent thinks, hops along a branch as work progresses (sprouting a leaf for each step), and fluffs up when the answer arrives.

Built with [three.js](https://threejs.org): layered shell-rendered plumage with procedurally painted markings, glossy eyes, and wings made from individually textured feathers.

## Files

- `ollie.js`: the mascot. One ES module, no other dependencies besides three.js.
- `index.html`: demo page with a state preview and an example chat.

## Run the demo

The demo uses ES modules, so serve it rather than opening the file directly:

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Use it in a project

With a bundler (Vite, webpack, Next.js and similar):

```bash
npm install three@^0.169
```

Then copy `ollie.js` into your source tree. Its `three` and `three/addons/...` imports resolve through the package.

Without a build step, add an import map to the page:

```html
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"
} }
</script>
```

## API

```js
import { OllieMascot } from './ollie.js';

// The canvas needs a CSS size; Ollie fills it.
const ollie = new OllieMascot(canvas, { mode: 'bar' }); // 'bar' (wide loading bar) or 'perch' (square avatar)

ollie.setState('thinking');   // no progress known yet
ollie.setProgress(0.4);       // 0..1; null returns to the indeterminate flying mode
ollie.setState('done');       // celebrates at the end of the branch
ollie.setState('idle');       // sits, blinks, follows the pointer (perch mode)
ollie.dispose();              // free GPU resources when the canvas goes away
```

Options: `mode`, `state`, `progress`, and `followPointer` (defaults to `true` in perch mode).

| State | Bar mode | Perch mode |
|---|---|---|
| `thinking` | Flies back and forth above the branch | Cocks its head side to side, pupils wide |
| `working` | Hops along the branch to the current progress | Looks down and flutters its wings |
| `done` | Reaches the end, fluffs up, stretches its wings | Same, then squints contentedly |
| `idle` | Sits on the branch | Follows the pointer and blinks |

### React example

```jsx
import { useEffect, useRef } from 'react';
import { OllieMascot } from './ollie.js';

export function OllieLoader({ state, progress }) {
  const ref = useRef(null);
  const ollie = useRef(null);
  useEffect(() => {
    ollie.current = new OllieMascot(ref.current, { mode: 'bar' });
    return () => ollie.current.dispose();
  }, []);
  useEffect(() => { ollie.current?.setState(state); }, [state]);
  useEffect(() => { if (progress != null) ollie.current?.setProgress(progress); }, [progress]);
  return <canvas ref={ref} style={{ width: 360, height: 104 }} aria-hidden="true" />;
}
```

## Notes

- Browser only. It needs WebGL and the DOM, so skip server-side rendering (in Next.js, load it with `dynamic(..., { ssr: false })`).
- Each instance uses its own WebGL context, and browsers cap these at roughly 16 per page. Keep one live owl on the message that is loading and `dispose()` it when the reply lands.
- Rendering pauses while the canvas is off screen. Small canvases use fewer feather layers automatically.
- Honors `prefers-reduced-motion` with slower, smaller movements.
