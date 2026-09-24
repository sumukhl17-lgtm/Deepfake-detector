const form = document.querySelector("#analyze-form");
const fileInput = document.querySelector("#image-file");
const fileName = document.querySelector("#file-name");
const analyzeButton = document.querySelector("#analyze-button");
const loadingMessage = document.querySelector("#loading-message");
const errorMessage = document.querySelector("#error-message");
const results = document.querySelector("#results");
const predictionLabel = document.querySelector("#prediction-label");
const confidenceValue = document.querySelector("#confidence-value");
const originalImage = document.querySelector("#original-image");
const heatmapImage = document.querySelector("#heatmap-image");
const forensicHeatmapImage = document.querySelector("#forensic-heatmap-image");
let originalObjectUrl = null;

fileInput.addEventListener("change", () => {
  const selectedFile = fileInput.files[0];
  fileName.textContent = selectedFile?.name || "JPG, JPEG, or PNG";
  if (originalObjectUrl) {
    URL.revokeObjectURL(originalObjectUrl);
    originalObjectUrl = null;
  }
  if (selectedFile) {
    originalObjectUrl = URL.createObjectURL(selectedFile);
    originalImage.src = originalObjectUrl;
  } else {
    originalImage.removeAttribute("src");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    showError("Please choose an image file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  setLoading(true);
  hideError();
  results.hidden = true;

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `Request failed (${response.status}).`);
    }
    renderResult(payload);
  } catch (error) {
    showError(error.message || "The analysis could not be completed.");
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  loadingMessage.hidden = !isLoading;
  analyzeButton.disabled = isLoading;
  analyzeButton.textContent = isLoading ? "Analyzing..." : "Analyze";
}

function renderResult(payload) {
  const label = String(payload.label || "").toUpperCase();
  predictionLabel.textContent = label;
  predictionLabel.className = `prediction-badge ${label.toLowerCase()}`;
  confidenceValue.textContent = `${(Number(payload.confidence) * 100).toFixed(1)}%`;
  heatmapImage.src = payload.heatmap_url;
  forensicHeatmapImage.src = payload.forensic_heatmap_url;
  results.hidden = false;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function hideError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}
