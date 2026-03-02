/* ── EventScout frontend app.js ────────────────────────────────────────── */

const CATEGORIES = [
    "🎵 Concerts & Live Music", "🏀 Sports", "🎭 Theater & Broadway",
    "😂 Comedy", "🎨 Arts & Exhibitions", "👨‍👩‍👧 Family & Kids",
    "🎉 Festivals & Fairs", "🍷 Food & Drink", "🎤 Hip-Hop & R&B",
    "🎸 Rock & Alternative", "🎷 Jazz & Blues", "💃 Dance & EDM",
    "🏛️ Cultural & Community", "🌿 Outdoor & Adventure",
];

let selectedCats = new Set(["🎵 Concerts & Live Music", "🏀 Sports"]);
let recommendations = null;   // last API response
let chatHistory = [];      // [{role, content}, …]

/* ── Init ─────────────────────────────────────────────────────────────── */
(function init() {
    // Default dates
    const today = new Date();
    const next7 = new Date(today); next7.setDate(today.getDate() + 7);
    document.getElementById("start-date").value = fmtDate(today);
    document.getElementById("end-date").value = fmtDate(next7);

    // Render category chips
    const wrap = document.getElementById("chips");
    CATEGORIES.forEach(cat => {
        const chip = document.createElement("div");
        chip.className = "chip" + (selectedCats.has(cat) ? " active" : "");
        chip.textContent = cat;
        chip.onclick = () => {
            if (selectedCats.has(cat)) { selectedCats.delete(cat); chip.classList.remove("active"); }
            else { selectedCats.add(cat); chip.classList.add("active"); }
        };
        wrap.appendChild(chip);
    });

    // Budget slider label
    const budgetSlider = document.getElementById("budget");
    const budgetDisplay = document.getElementById("budget-display");
    budgetSlider.addEventListener("input", () => {
        budgetDisplay.textContent = budgetSlider.value === "0" ? "Any" : `$${budgetSlider.value}`;
    });

    // Top-N slider label
    const topnSlider = document.getElementById("topn");
    const topnDisplay = document.getElementById("topn-display");
    topnSlider.addEventListener("input", () => {
        topnDisplay.textContent = topnSlider.value;
    });
})();

/* ── Search ───────────────────────────────────────────────────────────── */
async function search() {
    const city = document.getElementById("city").value.trim();
    const state = document.getElementById("state").value.trim();
    const country = document.getElementById("country").value.trim() || "US";
    const start = document.getElementById("start-date").value;
    const end = document.getElementById("end-date").value;
    const vibe = document.getElementById("vibe").value.trim();
    const budget = parseInt(document.getElementById("budget").value) || 0;
    const topN = parseInt(document.getElementById("topn").value) || 6;

    // Validation
    const errBox = document.getElementById("sidebar-error");
    errBox.style.display = "none";
    if (!vibe) { showError("Please fill in your vibe & preferences."); return; }
    if (!city) { showError("Please enter a city."); return; }
    if (start > end) { showError("Start date must be before end date."); return; }

    // Build event_description from chips + vibe
    const catLabels = [...selectedCats].map(c => c.split(" ").slice(1).join(" "));
    const eventDesc = catLabels.join(", ") + (vibe ? ". " + vibe : "");

    const payload = {
        city,
        state_code: state || null,
        country_code: country,
        start_date: start,
        end_date: end,
        event_description: eventDesc || "any events",
        venue_preference: "No preference",
        vibe_notes: vibe,
        budget_max: budget > 0 ? budget : null,
        selected_categories: [...selectedCats],
    };

    setLoading(true, "Finding relevant events…");

    try {
        const res = await fetch(`/api/recommend?top_n=${topN}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error ${res.status}`);
        }

        setLoading(true, "Ranking by your vibe…");
        recommendations = await res.json();
        chatHistory = [];

        renderResults(recommendations, city, start, end);
    } catch (e) {
        showError(e.message);
        setLoading(false);
    }
}

/* ── Render results ───────────────────────────────────────────────────── */
function renderResults(data, city, start, end) {
    setLoading(false);
    document.getElementById("hero").style.display = "none";
    document.getElementById("results").style.display = "block";
    document.getElementById("chat-section").style.display = "block";

    const recs = data.recommendations || [];

    // Stats bar
    const avgScore = recs.length
        ? (recs.reduce((s, r) => s + r.relevance_score, 0) / recs.length).toFixed(1)
        : "—";
    const weekendCount = recs.filter(r => r.event.is_weekend).length;
    document.getElementById("stats-bar").innerHTML = `
    <div class="stat-card"><div class="stat-label">Events Found</div><div class="stat-value">${recs.length}</div></div>
    <div class="stat-card"><div class="stat-label">Weekend Events</div><div class="stat-value">${weekendCount}</div></div>
    <div class="stat-card"><div class="stat-label">Avg Score</div><div class="stat-value">${avgScore}<span style="font-size:0.9rem;color:var(--muted)">/100</span></div></div>
  `;

    document.getElementById("results-title").textContent =
        `Top ${recs.length} Events in ${city} (${start} → ${end})`;

    const container = document.getElementById("cards-container");
    container.innerHTML = "";

    recs.forEach((rec, idx) => {
        const e = rec.event;
        const w = rec.weather;
        const score = Math.round(rec.relevance_score);

        const scoreClass = score >= 75 ? "score-green" : score >= 50 ? "score-yellow" : "score-red";

        const priceStr = (e.price_min || e.price_max)
            ? `$${Math.round(e.price_min)} – $${Math.round(e.price_max)}`
            : "Free / Not listed";

        const weatherStr = w
            ? `${w.description} · ${Math.round(w.temp_min_f)}–${Math.round(w.temp_max_f)}°F · Rain ${Math.round(w.precipitation_chance)}%`
            : "No forecast available";

        const timeStr = e.time && e.time !== "TBD" ? e.time.slice(0, 5) : "TBD";

        const imgHtml = e.image_url
            ? `<img class="event-img" src="${e.image_url}" alt="${esc(e.event_name)}" loading="lazy" />`
            : `<div class="event-img-placeholder">🎭</div>`;

        const ticketHtml = e.url
            ? `<a class="ticket-btn" href="${e.url}" target="_blank" rel="noopener">🎫 Get Tickets</a>`
            : "";

        const badgeList = [
            e.category ? `<span class="badge">${esc(e.category)}</span>` : "",
            e.genre ? `<span class="badge">${esc(e.genre)}</span>` : "",
            `<span class="badge">${e.is_outdoor ? "🌳 Outdoor" : "🏢 Indoor"}</span>`,
            `<span class="badge">${e.is_weekend ? "🗓️ Weekend" : "📅 Weekday"}</span>`,
        ].join("");

        const card = document.createElement("div");
        card.className = "event-card";
        card.innerHTML = `
      ${imgHtml}
      <div class="event-info">
        <div class="event-name">${idx + 1}. ${esc(e.event_name)}</div>
        <div class="badges">${badgeList}</div>
        <div class="event-meta">
          <div class="meta-item"><strong>📅 Date</strong>${e.date} @ ${timeStr}</div>
          <div class="meta-item"><strong>📍 Venue</strong>${esc(e.venue_name)}</div>
          <div class="meta-item"><strong>💵 Price</strong>${priceStr}</div>
          <div class="meta-item"><strong>🌤️ Weather</strong>${esc(weatherStr)}</div>
        </div>
        ${ticketHtml}
        <div class="event-reason">💡 ${esc(rec.score_reason)}</div>
      </div>
      <div class="score-col">
        <div class="score-badge ${scoreClass}">${score}</div>
        <div class="score-label">/ 100</div>
      </div>
    `;
        container.appendChild(card);
    });

    // Reset chat
    document.getElementById("chat-messages").innerHTML = "";
}

/* ── Chat ─────────────────────────────────────────────────────────────── */
async function sendChat() {
    if (!recommendations) return;
    const input = document.getElementById("chat-input");
    const q = input.value.trim();
    if (!q) return;

    input.value = "";
    appendMsg("user", q);
    chatHistory.push({ role: "user", content: q });

    const sendBtn = document.getElementById("chat-send");
    sendBtn.disabled = true;

    try {
        const payload = {
            recommendations,
            conversation_history: chatHistory.slice(0, -1), // send history minus current
            user_question: q,
        };

        const res = await fetch("/api/qa", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();

        appendMsg("assistant", data.answer);
        chatHistory = data.updated_history;
    } catch (e) {
        appendMsg("assistant", `Sorry, something went wrong: ${e.message}`);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

function appendMsg(role, text) {
    const box = document.getElementById("chat-messages");
    const div = document.createElement("div");
    div.className = `msg msg-${role}`;
    div.textContent = text;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

/* ── Helpers ──────────────────────────────────────────────────────────── */
function setLoading(on, text = "") {
    document.getElementById("loading").style.display = on ? "block" : "none";
    document.getElementById("results").style.display = on ? "none" : (recommendations ? "block" : "none");
    document.getElementById("hero").style.display = on || recommendations ? "none" : "block";
    document.getElementById("loading-text").textContent = text;
    document.getElementById("search-btn").disabled = on;
}

function showError(msg) {
    const box = document.getElementById("sidebar-error");
    box.textContent = msg;
    box.style.display = "block";
}

function fmtDate(d) {
    return d.toISOString().slice(0, 10);
}

function esc(str) {
    return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
