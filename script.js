// ==============================
// 🔧 Config
// ==============================
const API_BASE = "http://127.0.0.1:7860";
const BACKEND_FRAMES = `${API_BASE}/frames`;

const form = document.getElementById("searchForm");
const resultsGrid = document.getElementById("resultsGrid");
const downloadCsv = document.getElementById("downloadCsv");

// ==============================
// 🔍 SEARCH
// ==============================
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(form);

  // ✅ Thêm Audio Query (ASR)
  formData.append("audio_query", document.getElementById("audioQuery").value);

  resultsGrid.innerHTML = "<p>⏳ Đang tìm kiếm...</p>";

  try {
    const res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (data.status === "ok") {
      renderResults(data.results);
    } else {
      resultsGrid.innerHTML = `<p style="color:red;">❌ ${data.message}</p>`;
    }
  } catch (err) {
    console.error(err);
    resultsGrid.innerHTML = `<p style="color:red;">⚠️ ${err}</p>`;
  }
});

// ==============================
// 🖼 RENDER RESULTS
// ==============================
function renderResults(results) {
  resultsGrid.innerHTML = "";
  if (!results.length) {
    resultsGrid.innerHTML = "<p>Không có kết quả nào</p>";
    return;
  }

  results.forEach((r) => {
    const card = document.createElement("div");
    card.className = "result-card";

    // ✅ Lấy đường dẫn ảnh thật từ backend
    let imageUrl = r.url;
    if (!imageUrl) {
      let p = (r.path || "").replace(/^.*Videos_/, "Videos_");
      imageUrl = `${BACKEND_FRAMES}/${encodeURIComponent(p)}`;
    }

    // ✅ Hiển thị ảnh + thông tin OCR/ASR (nếu có)
    card.innerHTML = `
      <img class="frame-img"
           src="${imageUrl}" 
           alt="${r.frame_id || "Frame"}" 
           title="${r.path || ""}" 
           onerror="this.src='static/no_image.png';">
      <p><b>${r.frame_id || "Frame"}</b></p>
    `;

    const img = card.querySelector(".frame-img");

    // 🧲 Hiệu ứng kéo-thả
    img.addEventListener("dragover", (e) => {
      e.preventDefault();
      card.classList.add("drag-hover");
    });

    img.addEventListener("dragleave", (e) => {
      e.preventDefault();
      card.classList.remove("drag-hover");
    });

    img.addEventListener("drop", (e) => {
      e.preventDefault();
      card.classList.remove("drag-hover");

      const file = e.dataTransfer.files[0];
      if (!file || !file.type.startsWith("image/"))
        return alert("❌ Không phải ảnh hợp lệ!");

      const newURL = URL.createObjectURL(file);
      img.src = newURL;

      const nameElement = card.querySelector("p b");
      if (nameElement) {
        const newName = file.name.replace(/\.[^/.]+$/, "");
        nameElement.textContent = newName;
      }

      console.log(`🖼 Frame ${r.frame_id} đã được thay bằng: ${file.name}`);
    });

    // 🎯 Click mở context viewer
    card.addEventListener("click", () => openContext(r.frame_id));

    resultsGrid.appendChild(card);
  });
}

// ==============================
// 🧩 MỞ CONTEXT VIEWER (5x5 grid)
// ==============================
async function openContext(frameId) {
  const modal = document.createElement("div");
  modal.className = "modal";
  modal.innerHTML = `
  <div class="modal-content">
    <span class="close-btn">&times;</span>
    <h3>Context frames for <b>${frameId}</b></h3>

    <div class="submit-section">
      <h4>Submit to DRES</h4>
      <div class="submit-inputs">
        <input type="text" id="modalVideoId" placeholder="Video ID (e.g. K08_V001)">
        <input type="number" id="modalFrameId" placeholder="Frame ID" value="${frameId}">
        <input type="text" id="modalQaAnswer" placeholder="QA Answer">
        <input type="text" id="modalTrakeFrames" placeholder="Frame IDs (comma-separated)">
      </div>
      <div class="submit-buttons">
        <button id="modalSubmitKIS">Submit KIS</button>
        <button id="modalSubmitQA">Submit QA</button>
        <button id="modalSubmitTRAKE">Submit TRAKE</button>
      </div>
    </div>

    <div class="context-grid"></div>
  </div>
`;

  document.body.appendChild(modal);
  modal.querySelector(".close-btn").onclick = () => modal.remove();

  const grid = modal.querySelector(".context-grid");
  grid.innerHTML = "<p style='color:#ccc;'>Đang tải khung ảnh...</p>";

  try {
    const res = await fetch(`${API_BASE}/api/context/${frameId}`);
    const data = await res.json();

    if (data.status === "ok" && data.neighbors.length) {
      grid.innerHTML = "";
      const neighbors = data.neighbors;
      for (let i = 0; i < 25; i++) {
        const n = neighbors[i] || {};
        const cell = document.createElement("div");
        cell.className = "context-cell";

        let imgSrc = "static/no_image.png";
        if (n.path) {
          let p = n.path.replace(/^.*Videos_/, "Videos_");
          imgSrc = `${BACKEND_FRAMES}/${p}`;
        }

        cell.innerHTML = `
          <img src="${imgSrc}" alt="${n.frame_id || ""}">
          <small>${n.frame_id || ""}</small>
        `;
        if (n.frame_id === frameId) cell.classList.add("current");
        grid.appendChild(cell);
      }
    } else {
      grid.innerHTML = "<p style='color:red;'>Không tìm thấy frame lân cận.</p>";
    }
  } catch (err) {
    console.error(err);
    grid.innerHTML = `<p style="color:red;">⚠️ ${err}</p>`;
  }
}
  // ========== MAIN PAGE DRES SUBMIT ==========
const DRES_BASE = `${API_BASE}/dres`;

document.getElementById("submitKISMain").addEventListener("click", async () => {
  const video = document.getElementById("videoIdMain").value.trim();
  const frame = document.getElementById("frameIdMain").value.trim();
  if (!video || !frame) return alert("Missing video or frame ID.");

  const formData = new FormData();
  formData.append("videos_ID", video);
  formData.append("frame_start", frame);
  formData.append("frame_end", frame);

  try {
    const res = await fetch(`${DRES_BASE}/api/submit-kis`, { method: "POST", body: formData });
    const data = await res.json();
    alert("KIS submitted: " + JSON.stringify(data));
  } catch (err) {
    alert("Submit KIS failed: " + err);
  }
});

document.getElementById("submitQAMain").addEventListener("click", async () => {
  const video = document.getElementById("videoIdMain").value.trim();
  const frame = document.getElementById("frameIdMain").value.trim();
  const answer = document.getElementById("qaAnswerMain").value.trim();
  if (!video || !frame || !answer) return alert("Missing QA fields.");

  const formData = new FormData();
  formData.append("videos_ID", video);
  formData.append("frame_id", frame);
  formData.append("answer", answer);

  try {
    const res = await fetch(`${DRES_BASE}/api/submit-qa`, { method: "POST", body: formData });
    const data = await res.json();
    alert("QA submitted: " + JSON.stringify(data));
  } catch (err) {
    alert("Submit QA failed: " + err);
  }
});

document.getElementById("submitTRAKEMain").addEventListener("click", async () => {
  const video = document.getElementById("videoIdMain").value.trim();
  const frames = document.getElementById("trakeFramesMain").value.trim();
  if (!video || !frames) return alert("Missing TRAKE fields.");

  const formData = new FormData();
  formData.append("videos_ID", video);
  formData.append("frame_ids", frames);

  try {
    const res = await fetch(`${DRES_BASE}/api/submit-trake`, { method: "POST", body: formData });
    const data = await res.json();
    alert("TRAKE submitted: " + JSON.stringify(data));
  } catch (err) {
    alert("Submit TRAKE failed: " + err);
  }
});



// ==============================
// ⬇️ EXPORT CSV
// ==============================
downloadCsv.addEventListener("click", () => {
  const answer = document.getElementById("qaAnswer").value.trim();
  const cards = resultsGrid.querySelectorAll(".result-card");
  if (!cards.length) return alert("❌ Không có kết quả để export");

  const rows = Array.from(cards).map((c) => {
    const frame = c.querySelector("p").textContent.split(" ")[0];
    return answer ? [frame, answer] : [frame];
  });

  const csv = rows.map((r) => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "answers.csv";
  a.click();
});

// ==============================
// 🧩 DROPDOWN OBJECT FILTERS
// ==============================
async function loadObjectList() {
  try {
    const res = await fetch("object_list.csv");
    const text = await res.text();
    const lines = text.trim().split("\n");
    const list = document.getElementById("objectList");

    const hasHeader = lines[0].toLowerCase().includes("object");
    const dataLines = hasHeader ? lines.slice(1) : lines;

    dataLines.forEach(line => {
      const [objName, count] = line.split(",");
      if (!objName) return;
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${objName.trim()}"> ${objName.trim()} (${count || ""})`;
      list.appendChild(label);
    });

    console.log(`✅ Loaded ${dataLines.length} objects from CSV`);
  } catch (err) {
    console.error("❌ Failed to load object_list.csv:", err);
  }
}

const dropdown = document.querySelector(".dropdown");
const selected = document.getElementById("selectedObjects");
const list = document.getElementById("objectList");
const objectFiltersInput = document.getElementById("objectFilters");
const requireAllCheckbox = document.getElementById("requireAllCheckbox");
const requireAllInput = document.getElementById("requireAll");

selected.addEventListener("click", () => dropdown.classList.toggle("open"));
window.addEventListener("click", (e) => {
  if (!dropdown.contains(e.target)) dropdown.classList.remove("open");
});
list.addEventListener("change", () => {
  const checked = Array.from(list.querySelectorAll("input[type='checkbox']:checked"))
                       .map(cb => cb.value);
  selected.textContent = checked.length ? checked.join(", ") : "Select objects";
  objectFiltersInput.value = checked.join("\n");
});
requireAllCheckbox.addEventListener("change", () => {
  requireAllInput.value = requireAllCheckbox.checked ? "true" : "false";
});
document.addEventListener("DOMContentLoaded", loadObjectList);

// ==============================
// 💬 CHATBOT
// ==============================
const chatbotBtn = document.getElementById("chatbot-button");
const chatbotWindow = document.getElementById("chatbot-window");
const chatbotClose = document.getElementById("chatbot-close");
const chatbotBody = document.getElementById("chatbot-body");
const chatbotInput = document.getElementById("chatbot-input-text");
const chatbotSend = document.getElementById("chatbot-send");

chatbotBtn.addEventListener("click", () => chatbotWindow.classList.toggle("show"));
chatbotClose.addEventListener("click", () => chatbotWindow.classList.remove("show"));

chatbotSend.addEventListener("click", async () => {
  const msg = chatbotInput.value.trim();
  if (!msg) return;
  appendUserMessage(msg);
  chatbotInput.value = "";

  try {
    const res = await fetch(`${API_BASE}/api/chatbot`, {
      method: "POST",
      body: new URLSearchParams({ prompt: msg }),
    });
    const data = await res.json();
    appendBotMessage(data.reply || "⚠️ Không có phản hồi.");
  } catch (err) {
    appendBotMessage("❌ Lỗi gọi API Gemini");
  }
});

function appendUserMessage(text) {
  chatbotBody.innerHTML += `<div class="user-message">${text}</div>`;
}
function appendBotMessage(text) {
  chatbotBody.innerHTML += `<div class="bot-message">${text}</div>`;
  chatbotBody.scrollTop = chatbotBody.scrollHeight;
}

// ==============================
// 🚀 DRES SUBMISSION
// ==============================
