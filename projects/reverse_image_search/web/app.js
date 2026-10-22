/**
 * app.js: Interactive Client Logic for RCRE Lens Visual Studio
 */

let activeQueryFile = null;

document.addEventListener("DOMContentLoaded", () => {
    fetchSystemStatus();
    setupDropzones();
    runClustering(); // Auto-load initial visual clusters
});

/* Tab Switching */
function switchTab(tabName) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

    if (tabName === "search") {
        document.getElementById("tab-search-btn").classList.add("active");
        document.getElementById("view-search").classList.add("active");
    } else {
        document.getElementById("tab-clusters-btn").classList.add("active");
        document.getElementById("view-clusters").classList.add("active");
    }
}

/* System Status */
async function fetchSystemStatus() {
    try {
        const res = await fetch("/api/status");
        if (res.ok) {
            const data = await res.json();
            document.getElementById("stat-threads").innerText = `${data.cpu_cores} Cores`;
            document.getElementById("stat-indexed").innerText = `${data.indexed_images} imgs (${data.index_size_kb} KB)`;
        }
    } catch (e) {
        console.error("Status fetch error:", e);
    }
}

/* Drag & Drop Setup */
function setupDropzones() {
    const queryDropzone = document.getElementById("query-dropzone");
    const queryInput = document.getElementById("query-file-input");

    queryDropzone.addEventListener("click", (e) => {
        if (e.target.tagName !== "BUTTON" && !activeQueryFile) {
            queryInput.click();
        }
    });

    queryDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        queryDropzone.classList.add("dragover");
    });

    queryDropzone.addEventListener("dragleave", () => {
        queryDropzone.classList.remove("dragover");
    });

    queryDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        queryDropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleQueryFileSelected(e.dataTransfer.files[0]);
        }
    });

    queryInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleQueryFileSelected(e.target.files[0]);
        }
    });

    // Gallery batch upload dropzone
    const galleryDropzone = document.getElementById("gallery-dropzone");
    const galleryInput = document.getElementById("gallery-file-input");

    galleryDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        galleryDropzone.classList.add("dragover");
    });

    galleryDropzone.addEventListener("dragleave", () => {
        galleryDropzone.classList.remove("dragover");
    });

    galleryDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        galleryDropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            uploadGalleryFiles(e.dataTransfer.files);
        }
    });

    galleryInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadGalleryFiles(e.target.files);
        }
    });
}

function handleQueryFileSelected(file) {
    activeQueryFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById("query-preview-img").src = e.target.result;
        document.getElementById("query-dropzone-prompt").style.display = "none";
        document.getElementById("query-preview-container").style.display = "flex";
    };
    reader.readAsDataURL(file);
}

function clearQueryImage(e) {
    if (e) e.stopPropagation();
    activeQueryFile = null;
    document.getElementById("query-preview-img").src = "";
    document.getElementById("query-preview-container").style.display = "none";
    document.getElementById("query-dropzone-prompt").style.display = "block";
    document.getElementById("query-file-input").value = "";
    document.getElementById("results-grid").style.display = "none";
    document.getElementById("results-empty").style.display = "block";
    document.getElementById("search-metrics").style.display = "none";
}

/* Quick Demo Probe Loader */
async function loadSampleProbe(samplePath) {
    try {
        const response = await fetch("/" + samplePath);
        const blob = await response.blob();
        const file = new File([blob], samplePath.split("/").pop(), { type: blob.type || "image/bmp" });
        handleQueryFileSelected(file);
        // Automatically trigger search
        setTimeout(executeSearch, 200);
    } catch (e) {
        console.error("Error loading sample probe:", e);
    }
}

/* Execute Similarity Search */
async function executeSearch() {
    if (!activeQueryFile) {
        alert("Please select or drop a probe query image first!");
        return;
    }

    const topK = document.getElementById("topk-slider").value;
    const btn = document.getElementById("btn-run-search");
    btn.disabled = true;
    btn.innerHTML = `<span>⏳ Searching HNSW Index...</span>`;

    const formData = new FormData();
    formData.append("query", activeQueryFile);

    try {
        const res = await fetch(`/api/search?k=${topK}`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            alert(`Search failed: ${err.error || "Unknown error"}`);
            return;
        }

        const data = await res.json();
        renderSearchResults(data);
    } catch (e) {
        console.error("Search error:", e);
        alert(`Search error: ${e}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span>⚡ Search Similar Images</span>`;
    }
}

function renderSearchResults(data) {
    const results = data.results || [];
    const grid = document.getElementById("results-grid");
    const empty = document.getElementById("results-empty");
    const metrics = document.getElementById("search-metrics");

    if (results.length === 0) {
        grid.style.display = "none";
        empty.style.display = "block";
        metrics.style.display = "none";
        return;
    }

    empty.style.display = "none";
    grid.style.display = "grid";
    metrics.style.display = "flex";

    document.getElementById("metric-latency").innerText = `${data.search_time_ms.toFixed(3)} ms`;
    document.getElementById("metric-matches").innerText = `${results.length} Matches Found`;

    grid.innerHTML = "";
    results.forEach(item => {
        const simPct = (item.similarity * 100).toFixed(1);
        const card = document.createElement("div");
        card.className = "result-card";
        card.innerHTML = `
            <img class="result-thumb" src="${item.url}" alt="${item.filename}" onerror="this.src='https://via.placeholder.com/300x200?text=Image'">
            <div class="result-body">
                <div class="result-top">
                    <span class="rank-badge">#${item.rank} MATCH</span>
                    <span class="sim-score">${simPct}%</span>
                </div>
                <div class="sim-bar-bg">
                    <div class="sim-bar-fill" style="width: ${simPct}%"></div>
                </div>
                <div class="result-file" title="${item.filename}">
                    <strong>File:</strong> ${item.filename}
                </div>
                <div class="result-file">
                    <strong>L2 Distance:</strong> ${item.l2_distance.toFixed(4)}
                </div>
            </div>
        `;
        grid.appendChild(card);
    });
}

/* Visual Clustering (k-Means++) */
async function runClustering() {
    const k = document.getElementById("cluster-k-select").value;
    const spinner = document.getElementById("clusters-loading");
    const grid = document.getElementById("cluster-cards-grid");

    spinner.style.display = "block";
    grid.innerHTML = "";

    try {
        const res = await fetch(`/api/cluster?k=${k}`, { method: "POST" });
        if (!res.ok) {
            console.error("Cluster API error");
            return;
        }

        const data = await res.json();
        renderClusters(data);
    } catch (e) {
        console.error("Clustering error:", e);
    } finally {
        spinner.style.display = "none";
    }
}

function renderClusters(data) {
    const clusters = data.clusters || [];
    const grid = document.getElementById("cluster-cards-grid");
    grid.innerHTML = "";

    if (clusters.length === 0) {
        grid.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: var(--text-muted);">No images found in gallery database to cluster.</p>`;
        return;
    }

    clusters.forEach(c => {
        const card = document.createElement("div");
        card.className = "cluster-card";

        const imagesHtml = c.images.map(img => 
            `<img class="cluster-img-thumb" src="${img.url}" alt="${img.filename}" title="${img.filename}" onerror="this.style.display='none'">`
        ).join("");

        card.innerHTML = `
            <div class="cluster-header">
                <div>
                    <h3 class="cluster-title">Cluster ${c.cluster_id}</h3>
                    <p class="cluster-theme">🏷️ ${c.theme}</p>
                </div>
                <span class="cluster-count">${c.images.length} images</span>
            </div>
            <div class="cluster-images-grid">
                ${imagesHtml}
            </div>
        `;
        grid.appendChild(card);
    });
}

/* Upload Gallery Files */
async function uploadGalleryFiles(files) {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; ++i) {
        formData.append(`file_${i}`, files[i]);
    }

    const dropText = document.querySelector(".upload-gallery-box .text strong");
    dropText.innerText = `⏳ Uploading and Indexing ${files.length} images...`;

    try {
        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        if (res.ok) {
            dropText.innerText = `✓ Successfully added ${files.length} images to gallery!`;
            setTimeout(() => {
                dropText.innerText = "Add Images to Gallery Collection";
            }, 3000);
            fetchSystemStatus();
            runClustering();
        }
    } catch (e) {
        console.error("Gallery upload error:", e);
        dropText.innerText = "Upload failed!";
    }
}

/* Re-Index Gallery */
async function reindexGallery() {
    try {
        const res = await fetch("/api/index", { method: "POST" });
        if (res.ok) {
            alert("✓ Multi-core re-indexing completed successfully!");
            fetchSystemStatus();
            runClustering();
        }
    } catch (e) {
        console.error("Index error:", e);
    }
}
