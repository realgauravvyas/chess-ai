/* Chess pieces as inline SVG.
 *
 * The Unicode chess glyphs are unreliable: most system fonts draw them as
 * outline shapes with a transparent interior, so setting `color` tints only
 * the thin outline and a white piece and a black piece look identical. The
 * shapes below carry their own fill and stroke, which the stylesheet sets
 * per side, so contrast never depends on the visitor's fonts.
 *
 * fill and stroke are inherited SVG properties, so styling the <svg>
 * element cascades to every shape inside it.
 */
const PIECE_SVG = {
  p: `<circle cx="22.5" cy="13" r="5.2"/>
      <path d="M17.4 19.5h10.2c1 3.2 3.1 4.6 3.1 8.2H14.3c0-3.6 2.1-5 3.1-8.2z"/>
      <path d="M13.5 28.6h18v3.6h-18z"/>
      <path d="M11 32.6h23v4.4H11z"/>`,

  r: `<path d="M12 11.5h4.4v3h3.9v-3h4.4v3h3.9v-3H33V21H12z"/>
      <path d="M14.6 21h15.8l1.2 10.5H13.4z"/>
      <path d="M11.8 31.5h21.4v3.2H11.8z"/>
      <path d="M10.4 34.7h24.2V38H10.4z"/>`,

  n: `<path d="M13 37V29c0-6 3.5-9.5 7-11.5-1.6 1.4-4 1.2-5.3-.6-1.4-2-.5-4.8 1.4-6.4
              L18 8.6l2.6 2.2L23 7.4l1.2 3.4C29.6 12.6 33 19 33 27.5V37z"/>
      <circle cx="18.6" cy="15.4" r="1.15" class="detail"/>`,

  b: `<circle cx="22.5" cy="9.8" r="2.6"/>
      <path d="M22.5 12.4c5.4 3 7.6 9.2 5.4 14.2h-10.8c-2.2-5 0-11.2 5.4-14.2z"/>
      <path d="M16 26.6h13v3.2H16z"/>
      <path d="M13 29.8h19v3.4H13z"/>
      <path d="M10.8 33.2h23.4v3.6H10.8z"/>`,

  q: `<circle cx="9.6" cy="14.6" r="2.6"/><circle cx="16" cy="10.8" r="2.6"/>
      <circle cx="22.5" cy="9.4" r="2.8"/><circle cx="29" cy="10.8" r="2.6"/>
      <circle cx="35.4" cy="14.6" r="2.6"/>
      <path d="M10.4 16.6l3 12h18.2l3-12-5.4 6-3.6-8.4h-6.2L15.8 22.6z"/>
      <path d="M12.6 28.6h19.8v3.4H12.6z"/>
      <path d="M10.6 32h23.8v4.2H10.6z"/>`,

  k: `<path d="M21.2 5.6h2.6v3h3v2.6h-3v3.2h-2.6v-3.2h-3V8.6h3z"/>
      <path d="M22.5 15.4c4.6-4 12 0 12 6.2 0 3.4-2.4 6-5 8.2H15.5
              c-2.6-2.2-5-4.8-5-8.2 0-6.2 7.4-10.2 12-6.2z"/>
      <path d="M13 30h19v3.4H13z"/>
      <path d="M11 33.4h23V38H11z"/>`,
};

/* Build one piece element: <span class="piece w"><svg>…</svg></span> */
function pieceElement(type, colour) {
  const span = document.createElement("span");
  span.className = "piece " + colour;
  span.innerHTML =
    `<svg viewBox="0 0 45 45" width="100%" height="100%" ` +
    `aria-hidden="true">${PIECE_SVG[type]}</svg>`;
  return span;
}
