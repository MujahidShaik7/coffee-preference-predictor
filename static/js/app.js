/* ==============================================================
   Coffee Preference Predictor — Frontend logic
   Single-page state machine: landing -> quiz -> loading -> result
   ============================================================== */

(function () {
  "use strict";

  // ---------------------------------------------------------------
  // Question definitions
  // ---------------------------------------------------------------
  const QUESTIONS = [
    {
      key: "strength",
      type: "choice",
      title: "How strong do you like your coffee?",
      cols: 1,
      options: ["Light", "Medium", "Strong"],
    },
    {
      key: "milk",
      type: "choice",
      title: "How much milk do you usually add?",
      cols: 2,
      options: ["None", "Low", "Medium", "High"],
    },
    {
      key: "sweetness",
      type: "scale",
      title: "How sweet do you like your coffee?",
      labels: ["Not sweet", "Very sweet"],
    },
    {
      key: "temperature",
      type: "choice",
      title: "What temperature do you prefer?",
      cols: 1,
      options: ["Hot", "Iced", "Cold"],
    },
    {
      key: "bitterness",
      type: "scale",
      title: "How much bitterness can you handle?",
      labels: ["I avoid it", "Bring it on"],
    },
    {
      key: "creaminess",
      type: "scale",
      title: "How creamy do you like your coffee?",
      labels: ["Not creamy", "Very creamy"],
    },
    {
      key: "caffeine",
      type: "choice",
      title: "How much caffeine do you want?",
      cols: 1,
      options: ["Low", "Medium", "High"],
    },
    {
      key: "time_of_day",
      type: "choice",
      title: "When do you usually drink coffee?",
      cols: 2,
      options: ["Morning", "Afternoon", "Evening", "Night"],
    },
    {
      key: "flavor",
      type: "choice",
      title: "Which flavor profile appeals to you most?",
      cols: 2,
      options: ["Chocolate", "Nutty", "Fruity", "Caramel", "Classic", "Vanilla"],
    },
    {
      key: "experience",
      type: "choice",
      title: "What kind of coffee experience are you after?",
      cols: 1,
      options: ["Quick & Strong", "Smooth & Relaxing", "Sweet & Creamy", "Refreshing & Cold"],
    },
    {
      key: "previous",
      type: "choice",
      title: "Do you have a coffee you already love?",
      cols: 2,
      options: ["Espresso", "Cappuccino", "Latte", "Americano", "Mocha", "Cold Brew", "Filter Coffee", "None"],
    },
  ];

  const answers = {};
  let currentIndex = 0;

  // ---------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------
  const screens = {
    landing: document.getElementById("screen-landing"),
    quiz: document.getElementById("screen-quiz"),
    loading: document.getElementById("screen-loading"),
    result: document.getElementById("screen-result"),
    error: document.getElementById("screen-error"),
  };

  const btnStart = document.getElementById("btn-start");
  const btnBack = document.getElementById("btn-back");
  const btnContinue = document.getElementById("btn-continue");
  const quizCard = document.getElementById("quiz-card");
  const progressFill = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");
  const loadingText = document.getElementById("loading-text");
  const btnRetry = document.getElementById("btn-retry");
  const btnErrorRetry = document.getElementById("btn-error-retry");
  const btnShare = document.getElementById("btn-share");
  const brand = document.querySelector('[data-nav="landing"]');

  function showScreen(name) {
    Object.entries(screens).forEach(([key, el]) => {
      const active = key === name;
      el.classList.toggle("is-active", active);
      el.setAttribute("aria-hidden", active ? "false" : "true");
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ---------------------------------------------------------------
  // Quiz rendering
  // ---------------------------------------------------------------
  function renderQuestion() {
    const q = QUESTIONS[currentIndex];
    const total = QUESTIONS.length;

    progressLabel.textContent = `Question ${currentIndex + 1} of ${total}`;
    progressFill.style.width = `${((currentIndex) / total) * 100 + (100 / total) * 0.15}%`;
    btnBack.disabled = currentIndex === 0;

    let html = `<p class="q-eyebrow">Preference</p><h2 class="q-title">${q.title}</h2>`;

    if (q.type === "choice") {
      html += `<div class="option-grid cols-${q.cols}" role="group" aria-label="${q.title}">`;
      q.options.forEach((opt) => {
        const selected = answers[q.key] === opt;
        html += `
          <button type="button" class="option-btn${selected ? " is-selected" : ""}" data-value="${opt}">
            <span>${opt}</span>
            <span class="check" aria-hidden="true">
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none"><path d="M2 6L5 9L10 3" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </span>
          </button>`;
      });
      html += `</div>`;
    } else if (q.type === "scale") {
      html += `<div class="scale-row" role="radiogroup" aria-label="${q.title}">`;
      for (let i = 1; i <= 5; i++) {
        const selected = answers[q.key] === i;
        html += `<button type="button" class="scale-btn${selected ? " is-selected" : ""}" data-value="${i}" role="radio" aria-checked="${selected}">${i}</button>`;
      }
      html += `</div><div class="scale-labels"><span>${q.labels[0]}</span><span>${q.labels[1]}</span></div>`;
    }

    quizCard.innerHTML = html;
    quizCard.querySelectorAll("[data-value]").forEach((btn) => {
      btn.addEventListener("click", () => selectAnswer(q, btn));
    });

    btnContinue.disabled = answers[q.key] === undefined;
    btnContinue.textContent = "";
    btnContinue.innerHTML =
      currentIndex === total - 1
        ? `Predict My Coffee`
        : `Continue
           <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M3 8H13M13 8L9 4M13 8L9 12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }

  function selectAnswer(question, btnEl) {
    const raw = btnEl.getAttribute("data-value");
    const value = question.type === "scale" ? parseInt(raw, 10) : raw;
    answers[question.key] = value;

    quizCard.querySelectorAll("[data-value]").forEach((b) => {
      const isThis = b === btnEl;
      b.classList.toggle("is-selected", isThis);
      if (b.getAttribute("role") === "radio") b.setAttribute("aria-checked", isThis ? "true" : "false");
    });

    btnContinue.disabled = false;
  }

  function goNext() {
    if (answers[QUESTIONS[currentIndex].key] === undefined) return;
    if (currentIndex === QUESTIONS.length - 1) {
      runPrediction();
      return;
    }
    currentIndex += 1;
    renderQuestion();
  }

  function goBack() {
    if (currentIndex === 0) return;
    currentIndex -= 1;
    renderQuestion();
  }

  // keyboard accessibility: left/right arrows navigate when possible
  document.addEventListener("keydown", (e) => {
    if (!screens.quiz.classList.contains("is-active")) return;
    if (e.key === "ArrowRight" && !btnContinue.disabled) goNext();
    if (e.key === "ArrowLeft" && !btnBack.disabled) goBack();
  });

  // ---------------------------------------------------------------
  // Prediction flow
  // ---------------------------------------------------------------
  const LOADING_MESSAGES = [
    "Analyzing your coffee preferences…",
    "Comparing against thousands of profiles…",
    "Finding your perfect match…",
  ];

  async function runPrediction() {
    showScreen("loading");
    let step = 0;
    loadingText.textContent = LOADING_MESSAGES[0];
    const interval = setInterval(() => {
      step = (step + 1) % LOADING_MESSAGES.length;
      loadingText.textContent = LOADING_MESSAGES[step];
    }, 550);

    const minWait = new Promise((res) => setTimeout(res, 900));

    try {
      const [response] = await Promise.all([
        fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(answers),
        }),
        minWait,
      ]);

      clearInterval(interval);

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.error || "Prediction request failed.");
      }

      const data = await response.json();
      renderResult(data);
      showScreen("result");
    } catch (err) {
      clearInterval(interval);
      showError(err.message);
    }
  }

  function renderResult(data) {
    document.getElementById("result-name").textContent = data.prediction;
    document.getElementById("result-tagline").textContent = data.tagline || "";
    document.getElementById("result-desc").textContent = data.description || "";
    document.getElementById("personality-name").textContent = data.personality || "";
    document.getElementById("personality-desc").textContent = data.personality_description || "";

    const pct = Math.round((data.confidence || 0) * 100);
    document.getElementById("result-confidence").textContent = `${pct}%`;

    const matchWrap = document.getElementById("result-match");
    matchWrap.innerHTML = `<span class="match-percent">${pct}%</span> Match`;

    const reasonsList = document.getElementById("reasons-list");
    reasonsList.innerHTML = "";
    (data.why_this_matches_you || []).forEach((reason) => {
      const li = document.createElement("li");
      li.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M3 8.5L6.2 11.5L13 4.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span>${escapeHTML(reason)}</span>`;
      reasonsList.appendChild(li);
    });

    // if confidence is on the lower side, soften the framing
    const descEl = document.getElementById("result-desc");
    if (pct < 45) {
      descEl.textContent =
        (data.description || "") +
        " You have a few strong coffee matches — this one came out on top.";
    }

    btnShare.onclick = () => shareResult(data.prediction, pct);
  }

  function shareResult(coffee, pct) {
    const text = `☕ My coffee match is ${coffee} — ${pct}% match!`;
    if (navigator.share) {
      navigator.share({ text, title: "My Coffee Match" }).catch(() => {});
    } else if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(() => {
        btnShare.textContent = "Copied!";
        setTimeout(() => (btnShare.textContent = "Share Result"), 1600);
      });
    }
  }

  function showError(message) {
    document.getElementById("error-desc").textContent =
      message && message.length < 140
        ? message
        : "Something went wrong while finding your coffee match. Please try again.";
    showScreen("error");
  }

  function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ---------------------------------------------------------------
  // Navigation wiring
  // ---------------------------------------------------------------
  function startQuiz() {
    currentIndex = 0;
    Object.keys(answers).forEach((k) => delete answers[k]);
    showScreen("quiz");
    renderQuestion();
  }

  btnStart.addEventListener("click", startQuiz);
  btnContinue.addEventListener("click", goNext);
  btnBack.addEventListener("click", goBack);
  btnRetry.addEventListener("click", startQuiz);
  btnErrorRetry.addEventListener("click", startQuiz);
  brand.addEventListener("click", () => showScreen("landing"));
  brand.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      showScreen("landing");
    }
  });
})();
