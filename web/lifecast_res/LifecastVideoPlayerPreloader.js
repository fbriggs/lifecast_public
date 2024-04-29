export function init({
  _embed_in_div = "",
}={}) {
  let container = document.getElementById(_embed_in_div);
  if (_embed_in_div == "") {
    container = document.body;
  }

  const buffering_button = document.createElement("img");
  buffering_button.id                       = "lifecast_preload_buffering_button";
  buffering_button.className                = "lifecast_preload_indicator";
  buffering_button.src                      = "lifecast_res/spinner.png";
  buffering_button.draggable                = false;
  buffering_button.style.opacity            = .5;
  buffering_button.style.width              = "64px";
  buffering_button.style.height             = "64px";
  buffering_button.style.position           = 'absolute';

  if (_embed_in_div == "") {
    buffering_button.style.top                = "50vh";
  } else {
    buffering_button.style.top                = '50%';
  }

  buffering_button.style.left               = '50%';
  buffering_button.style.transform          = 'translate(-50%, -50%)';
  buffering_button.style.animation          = 'spin 2s linear infinite';

  const styleTag = document.createElement("style");
  styleTag.textContent = `
    @keyframes spin {
      from {
        transform: translate(-50%, -50%) rotate(0deg);
      }
      to {
        transform: translate(-50%, -50%) rotate(360deg);
      }
    }
  `;

  document.head.appendChild(styleTag);

  container.appendChild(buffering_button);

  // Ensure the spinner centers within the container by adjusting its CSS
  container.style.position = "relative"; // This keeps the spinner centered within the container
}
