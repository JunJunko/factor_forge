const $ = (id) => document.getElementById(id);

const PIPELINE_STEPS = [
  ["update_data", "更新数据"],
  ["generate_all_model_signals", "生成模型信号"],
  ["compute_health", "计算健康度"],
  ["decide_position_state", "决定仓位状态"],
  ["build_orders", "生成订单草稿"],
  ["execution_audit", "执行审计"],
  ["shadow_report", "影子报告"],
];

let pollTimer = null;
let latestSignal = null;

function todayShanghai() {
  const now = new Date();
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(now);
  const get = (type) => parts.find((p) => p.type === type).value;
  return `${get("year")}-${get("month")}-${get("day")}`;
}

function ymdCompact(dateText) {
  return dateText.replaceAll("-", "");
}

function fmtPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  return `${(n * 100).toFixed(2)}%`;
}

function fmtNum(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  return n.toFixed(digits);
}

function fmtMoney(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  if (Math.abs(n) >= 1e8) return `${(n / 1e8).toFixed(2)}亿`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e4).toFixed(2)}万`;
  return n.toFixed(0);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderKv(el, rows) {
  el.innerHTML = rows.map(([k, v, help]) => {
    const hint = help
      ? `<span class="helpWrap">
          <button class="helpDot" type="button" aria-label="${escapeHtml(k)}说明" aria-expanded="false">?</button>
          <span class="helpTip" role="tooltip">${escapeHtml(help)}</span>
        </span>`
      : "";
    return `<dt><span>${escapeHtml(k)}</span>${hint}</dt><dd>${v ?? ""}</dd>`;
  }).join("");
}

function closeHelpTips(exceptButton = null) {
  document.querySelectorAll(".helpDot.isOpen").forEach((button) => {
    if (button === exceptButton) return;
    button.classList.remove("isOpen");
    button.setAttribute("aria-expanded", "false");
  });
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function renderPipeline(activeStep = "", done = new Set(), failed = false) {
  $("pipeline").innerHTML = PIPELINE_STEPS.map(([key, label], idx) => {
    const cls = failed && key === activeStep ? "failed" : key === activeStep ? "active" : done.has(key) ? "done" : "";
    return `
      <div class="pipeStep ${cls}">
        <span>${idx + 1}</span>
        <strong>${label}</strong>
        <em>${key}</em>
      </div>
    `;
  }).join("");
}

function pipelineFromLogs(logs, status) {
  const done = new Set();
  let active = "";
  for (const [key] of PIPELINE_STEPS) {
    if (logs.some((line) => line.includes(`[${key}]`))) active = key;
    if (logs.some((line) => line.includes(`[${key}] done`) || line.includes(`[${key}] rows=`) || line.includes(`[${key}] blocking_trade_issues=`) || line.includes(`[${key}] overall=`) || line.includes(`[${key}] state=`))) {
      done.add(key);
    }
  }
  if (status === "succeeded") {
    PIPELINE_STEPS.forEach(([key]) => done.add(key));
    active = "";
  }
  renderPipeline(active, done, status === "failed");
}

function renderSignal(signal) {
  latestSignal = signal || null;
  const topRows = $("topRows");
  const banner = $("exposureBanner");
  const links = $("fileLinks");
  topRows.innerHTML = "";
  links.innerHTML = "";
  banner.textContent = "等待信号。";
  if (!signal || !signal.summary) return;

  const s = signal.summary;
  const exposure = Number(s.final_exposure ?? 0);
  banner.textContent = exposure <= 0
    ? `信号日 ${String(s.signal_date).slice(0, 10)}：目标仓位 0%，只观察不建仓。`
    : `信号日 ${String(s.signal_date).slice(0, 10)}：目标仓位 ${fmtPct(exposure)}，下一交易日开盘执行。`;

  topRows.innerHTML = (signal.top || []).map((row) => `
    <tr>
      <td><input class="candidateCheck" type="checkbox" value="${row.ts_code ?? ""}" /></td>
      <td>${row.rank ?? ""}</td>
      <td>${row.ts_code ?? ""}</td>
      <td>${row.name ?? ""}</td>
      <td>${row.industry_l1_name ?? ""}</td>
      <td>${fmtNum(row.factor_value, 4)}</td>
      <td>${fmtPct(row.target_weight)}</td>
      <td>${fmtNum(row.raw_close, 2)}</td>
      <td>${fmtMoney(row.amount_cny)}</td>
    </tr>
  `).join("");

  links.innerHTML = Object.entries(signal.files || {})
    .map(([name, path]) => `<a href="/api/file?path=${encodeURIComponent(path)}" target="_blank">${name}</a>`)
    .join("");
}

function selectedCandidateCodes() {
  return Array.from(document.querySelectorAll(".candidateCheck:checked"))
    .map((el) => el.value)
    .filter(Boolean);
}

function renderSellAdvice(data) {
  const banner = $("sellBanner");
  const rows = $("sellRows");
  rows.innerHTML = "";
  banner.classList.add("hidden");
  banner.classList.remove("flat");
  if (!data || !data.items || data.items.length === 0) {
    banner.classList.remove("hidden");
    banner.classList.add("flat");
    banner.textContent = "没有持仓输入。";
    return;
  }
  const exposure = Number(data.final_exposure);
  banner.classList.remove("hidden");
  banner.textContent = Number.isFinite(exposure) && exposure <= 0
    ? `信号日 ${data.signal_date}：组合风控仓位为 0%，所有持仓优先给出卖出建议。`
    : `信号日 ${data.signal_date}：按持有天数、组合风控和卖压避雷逐项检查。`;

  rows.innerHTML = data.items.map((row) => {
    const isSell = row.action === "SELL";
    return `
      <tr>
        <td>${row.ts_code ?? ""}</td>
        <td>${row.name ?? ""}</td>
        <td>${row.entry_date ?? ""}</td>
        <td>${row.holding_trade_days ?? ""}</td>
        <td><span class="badge ${isSell ? "sell" : "hold"}">${row.action ?? ""}</span></td>
        <td class="reason">${row.reason ?? ""}</td>
        <td>${fmtNum(row.sell_impact_efficiency, 4)}</td>
        <td>${fmtNum(row.sell_impact_deviation_60d, 4)}</td>
        <td>${row.hazard_strict ? "触发" : ""}</td>
      </tr>
    `;
  }).join("");
}

function renderWarnings(warnings) {
  const el = $("dailyWarnings");
  const rows = warnings || [];
  el.classList.toggle("hidden", rows.length === 0);
  el.innerHTML = rows.map((item) => `<div>${item}</div>`).join("");
}

function renderDashboard(dashboard) {
  if (!dashboard) return;
  const execution = dashboard.execution || {};
  const risk = dashboard.risk || {};
  const fit = risk.fit_quality || {};
  const gate = risk.risk_gate_inputs || {};
  const audit = dashboard.research_audit || {};
  const tradeAudit = audit.trade_audit || {};
  const latestYear = audit.frozen_latest_year || {};
  const sensitivity = audit.frozen_sensitivity || {};

  $("decisionBadge").textContent = dashboard.status || "-";
  $("decisionBadge").className = `decisionBadge ${dashboard.status === "观察" ? "watch" : "ok"}`;
  $("decisionText").textContent = execution.final_exposure <= 0
    ? "当前主决策是不建仓，只保留候选观察和持仓卖出检查。"
    : "当前允许按目标仓位生成订单草稿，执行前仍需确认次日开盘可成交。";
  renderWarnings(dashboard.warnings);

  renderKv($("executionSummary"), [
    ["信号日", execution.signal_date, "这批候选股使用哪一天收盘后可得的数据生成。实盘只能在之后的交易日执行。"],
    ["执行口径", execution.intended_execution === "next_trade_day_open" ? "下一交易日开盘" : execution.intended_execution, "确认是否严格用下一交易日开盘成交，不用信号日收盘价。"],
    ["最终仓位", fmtPct(execution.final_exposure), "今日决策的最终资金暴露。0% 表示不新建仓，只观察或处理已有持仓。"],
    ["目标持仓", `${execution.target_position_count ?? ""} / ${execution.candidate_count ?? ""}`, "前者是实际分配权重的股票数，后者是页面展示的候选数。"],
    ["信号阻塞", execution.signal_day_block_count, "候选股在信号日已经发现的停牌、ST、涨停开盘等风险标记数量。执行前还要看下一交易日。"],
  ]);

  renderKv($("riskSummary"), [
    ["Payoff门控", fmtPct(risk.risk_gate), "策略自身最近已完成 Top5 批次支持的仓位比例。越低越说明近期收益证据不足。"],
    ["10日净收益", fmtPct(gate.payoff_mean_net_10d), "最近窗口内 Top5 批次持有10日、扣20bps成本后的平均收益。"],
    ["10日LCB", fmtPct(gate.payoff_lcb_net_10d), "保守下界。为负表示均值虽可能为正，但稳定性还不够，仓位应打折。"],
    ["有效样本", fmtNum(gate.payoff_effective_obs, 1), "按10日持有重叠折算后的有效样本数，比原始日度样本更接近真实独立观察数。"],
    ["HMM仓位", fmtPct(risk.hmm_exposure), "市场状态模块给出的仓位。为0时，即使候选股有分数也不新开仓。"],
    ["fit方向", Number(fit.score_direction) < 0 ? "反向" : "正向", "冻结规则判断模型近期排序方向。反向表示近期高分组表现弱于低分组，分数已反向使用。"],
  ]);

  renderKv($("auditSummary"), [
    ["执行红灯", tradeAudit.blocking_trade_issues ?? "", "交易审计发现的硬错误数量。大于0时不要直接按订单执行。"],
    ["最近年收益", latestYear.year ? `${latestYear.year}: ${fmtPct(latestYear.return)}` : "", "冻结策略在最近测试年度的绝对收益，用来感知最近环境是否友好。"],
    ["最近年超额", fmtPct(latestYear.excess_return), "相对中证1000基准的最近年度超额。比绝对收益更适合判断策略是否真有贡献。"],
  ]);

  $("auditLinks").innerHTML = (audit.files || [])
    .map((item) => `<a href="/api/file?path=${encodeURIComponent(item.path)}" target="_blank">${item.label}</a>`)
    .join("");
}

function renderDailyResult(result) {
  if (!result) return;
  const state = result.position_state || {};
  const health = result.health || {};
  const audit = result.execution_audit || {};
  $("decisionBadge").textContent = state.state || "-";
  $("decisionBadge").className = `decisionBadge ${state.state === "FLAT" || state.state === "OBSERVE" ? "watch" : "ok"}`;
  $("decisionText").textContent = `${state.reason || ""} 健康度：${health.overall || "-"}；执行红灯：${audit.blocking_trade_issues ?? "-"}`;
  const files = result.files || {};
  $("auditLinks").innerHTML = Object.entries(files)
    .map(([name, path]) => `<a href="/api/file?path=${encodeURIComponent(path)}" target="_blank">${name}</a>`)
    .join("");
}

function renderShadowPortfolio(data) {
  const summary = data?.summary || {};
  const rows = data?.positions || [];
  const text = rows.length
    ? `latest ${String(summary.latest_date || "").slice(0, 10)}；持仓 ${summary.position_count ?? 0}，卖出建议 ${summary.sell_count ?? 0}，已评估 ${summary.evaluated_count ?? 0}，待入场 ${summary.pending_count ?? 0}，阻塞 ${summary.blocked_count ?? 0}，平均收益 ${fmtPct(summary.avg_shadow_return)}，胜率 ${fmtPct(summary.win_rate)}`
    : "尚未加入影子持仓。";
  $("shadowSummary").textContent = text;
  $("shadowRows").innerHTML = rows.map((row) => {
    const ret = Number(row.shadow_return);
    const cls = Number.isFinite(ret) && ret < 0 ? "neg" : Number.isFinite(ret) && ret > 0 ? "pos" : "";
    const id = String(row.id ?? "");
    const isClosed = String(row.status ?? "OPEN") === "CLOSED";
    const sellAction = row.sell_action || "";
    const isSell = sellAction === "SELL";
    const actions = id
      ? `<div class="rowActions">
          ${isClosed ? `<span class="mutedMini">已卖出</span>` : `<button class="miniBtn sellBtn" data-shadow-action="sell" data-position-id="${id}" onclick="handleShadowAction(event, 'sell', '${id}')">卖出</button>`}
          <button class="miniBtn dangerBtn" data-shadow-action="delete" data-position-id="${id}" onclick="handleShadowAction(event, 'delete', '${id}')">删除</button>
        </div>`
      : "";
    return `
      <tr>
        <td>${row.ts_code ?? ""}</td>
        <td>${row.name ?? ""}</td>
        <td>${String(row.signal_date ?? "").slice(0, 10)}</td>
        <td>${String(row.entry_date ?? "").slice(0, 10)}</td>
        <td><span class="statusPill">${row.eval_status ?? row.status ?? ""}</span></td>
        <td>${row.holding_trade_days ?? ""}</td>
        <td>${fmtNum(row.entry_raw_open, 2)}</td>
        <td>${fmtNum(row.mark_raw_close, 2)}</td>
        <td class="${cls}">${fmtPct(row.shadow_return)}</td>
        <td><span class="badge ${isSell ? "sell" : "hold"}">${sellAction}</span></td>
        <td class="reason">${row.sell_reason ?? ""}</td>
        <td>${fmtNum(row.sell_impact_efficiency, 4)}</td>
        <td>${fmtNum(row.sell_impact_deviation_60d, 4)}</td>
        <td>${row.hazard_strict ? "触发" : ""}</td>
        <td>${actions}</td>
      </tr>
    `;
  }).join("");
}

async function updateShadowPosition(positionId, action) {
  if (!positionId) return;
  if (action === "delete" && !confirm("删除这条影子持仓？此操作用于清理误触记录。")) return;
  const data = await api(`/api/shadow-portfolio/${action}`, {
    method: "POST",
    body: JSON.stringify({ position_id: positionId }),
  });
  renderShadowPortfolio(data.portfolio);
  $("taskMeta").textContent = action === "delete" ? "已删除影子持仓。" : "已卖出影子持仓。";
}

function handleShadowAction(event, action, positionId) {
  event.preventDefault();
  event.stopPropagation();
  setBusy(true);
  updateShadowPosition(positionId, action)
    .catch((err) => {
      $("taskMeta").textContent = String(err);
      $("logs").textContent = String(err);
    })
    .finally(() => setBusy(false));
}

async function refreshShadowPortfolio() {
  const data = await api("/api/shadow-portfolio");
  renderShadowPortfolio(data);
}

async function addSelectedToShadow() {
  const codes = selectedCandidateCodes();
  if (!codes.length) throw new Error("请先勾选候选股。");
  const summary = latestSignal?.summary;
  const signalDate = summary?.signal_date ? String(summary.signal_date).slice(0, 10) : $("signalDate").value;
  const data = await api("/api/shadow-portfolio/add", {
    method: "POST",
    body: JSON.stringify({ signal_date: signalDate, ts_codes: codes }),
  });
  renderShadowPortfolio(data.portfolio);
  $("taskMeta").textContent = `已加入影子持仓：${(data.added || []).length} 个`;
}

async function refreshStatus() {
  const data = await api("/api/status");
  const timing = data.data?.timing_position || {};
  renderKv($("dataStatus"), [
    ["版本", data.data?.version],
    ["范围", `${data.data?.start_date ?? ""} ~ ${data.data?.end_date ?? ""}`],
    ["行数", data.data?.row_count?.toLocaleString?.() ?? data.data?.row_count],
    ["完整面板", data.data?.complete ? "是" : "否"],
    ["Timing date", timing.latest_date ? String(timing.latest_date).slice(0, 10) : ""],
    ["Timing position", Number.isFinite(Number(timing.target_position)) ? fmtPct(timing.target_position) : ""],
  ]);
  const summary = data.latest_signal?.summary;
  const defaultSignalDate = data.default_signal_date || data.data?.end_date || (summary?.signal_date ? String(summary.signal_date).slice(0, 10) : "");
  if (defaultSignalDate) {
    $("signalDate").value = String(defaultSignalDate).slice(0, 10);
  }
  renderKv($("signalStatus"), [
    ["目录", data.latest_signal?.run_dir],
    ["信号日", summary?.signal_date ? String(summary.signal_date).slice(0, 10) : ""],
    ["算法", summary?.signal_algorithm || summary?.model],
    ["排序方向", summary?.fit_quality ? (Number(summary.fit_quality.score_direction) < 0 ? "反向" : "正向") : ""],
    ["最终仓位", summary ? fmtPct(summary.final_exposure) : ""],
  ]);
  renderDashboard(data.dashboard);
  renderSignal(data.latest_signal);
  await refreshShadowPortfolio().catch(() => {});
}

function setBusy(isBusy) {
  $("dailyRunBtn").disabled = isBusy;
  $("syncBtn").disabled = isBusy;
  $("signalBtn").disabled = isBusy;
  $("sellAdviceBtn").disabled = isBusy;
  $("addShadowBtn").disabled = isBusy;
  $("refreshShadowBtn").disabled = isBusy;
  document.querySelectorAll("[data-shadow-action]").forEach((button) => {
    button.disabled = isBusy;
  });
}

function startPolling(taskId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => pollTask(taskId), 1200);
  pollTask(taskId);
}

async function pollTask(taskId) {
  const task = await api(`/api/tasks/${taskId}`);
  const logs = task.logs || [];
  $("taskMeta").textContent = `任务 ${task.id}：${task.status}`;
  $("logs").textContent = logs.join("\n");
  $("logs").scrollTop = $("logs").scrollHeight;
  pipelineFromLogs(logs, task.status);
  if (task.status === "succeeded" || task.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
    setBusy(false);
    if (task.status === "succeeded") {
      await refreshStatus();
      renderDailyResult(task.result);
    }
  }
}

async function runDailyChain() {
  const signalDate = $("signalDate").value;
  const payload = {
    signal_date: signalDate,
    update_data: $("runUpdateData").checked,
    force_regenerate_signals: $("forceRegenerate").checked,
    start: $("syncStart").value.trim() || ymdCompact(signalDate),
    end: $("syncEnd").value.trim() || ymdCompact(signalDate),
    merge_full_history: $("mergeFull").checked,
    holdings_text: $("holdingsText").value,
  };
  localStorage.setItem("factorForgeHoldings", payload.holdings_text);
  setBusy(true);
  $("logs").textContent = "";
  renderPipeline();
  const res = await api("/api/daily-chain", { method: "POST", body: JSON.stringify(payload) });
  startPolling(res.task_id);
}

async function runSync() {
  const payload = {
    start: $("syncStart").value.trim(),
    end: $("syncEnd").value.trim(),
    merge_full_history: $("mergeFull").checked,
  };
  setBusy(true);
  $("logs").textContent = "";
  const res = await api("/api/sync", { method: "POST", body: JSON.stringify(payload) });
  startPolling(res.task_id);
}

async function runSignal() {
  const payload = { signal_date: $("signalDate").value };
  setBusy(true);
  $("logs").textContent = "";
  const res = await api("/api/signal", { method: "POST", body: JSON.stringify(payload) });
  startPolling(res.task_id);
}

async function runSellAdvice() {
  const holdings = $("holdingsText").value;
  localStorage.setItem("factorForgeHoldings", holdings);
  const payload = { signal_date: $("signalDate").value, holdings_text: holdings };
  const data = await api("/api/sell-advice", { method: "POST", body: JSON.stringify(payload) });
  renderSellAdvice(data);
}

function initDates() {
  const today = todayShanghai();
  $("signalDate").value = today;
  $("syncStart").value = ymdCompact(today);
  $("syncEnd").value = ymdCompact(today);
  $("holdingsText").value = localStorage.getItem("factorForgeHoldings") || "";
  renderPipeline();
}

$("refreshBtn").addEventListener("click", refreshStatus);
$("dailyRunBtn").addEventListener("click", () => runDailyChain().catch((err) => {
  setBusy(false);
  $("taskMeta").textContent = "每日主链路提交失败";
  $("logs").textContent = String(err);
}));
$("syncBtn").addEventListener("click", () => runSync().catch((err) => {
  setBusy(false);
  $("taskMeta").textContent = "同步任务提交失败";
  $("logs").textContent = String(err);
}));
$("signalBtn").addEventListener("click", () => runSignal().catch((err) => {
  setBusy(false);
  $("taskMeta").textContent = "信号任务提交失败";
  $("logs").textContent = String(err);
}));
$("sellAdviceBtn").addEventListener("click", () => runSellAdvice().catch((err) => {
  $("sellBanner").classList.remove("hidden");
  $("sellBanner").classList.add("flat");
  $("sellBanner").textContent = String(err);
}));
$("addShadowBtn").addEventListener("click", () => addSelectedToShadow().catch((err) => {
  $("taskMeta").textContent = "加入影子持仓失败";
  $("logs").textContent = String(err);
}));
$("refreshShadowBtn").addEventListener("click", () => refreshShadowPortfolio().catch((err) => {
  $("taskMeta").textContent = "影子评估刷新失败";
  $("logs").textContent = String(err);
}));

document.addEventListener("click", (event) => {
  const helpButton = event.target.closest(".helpDot");
  if (helpButton) {
    const nextOpen = !helpButton.classList.contains("isOpen");
    closeHelpTips(helpButton);
    helpButton.classList.toggle("isOpen", nextOpen);
    helpButton.setAttribute("aria-expanded", String(nextOpen));
    return;
  }
  if (!event.target.closest(".helpWrap")) {
    closeHelpTips();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeHelpTips();
});

$("shadowRows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-shadow-action]");
  if (!button || button.disabled) return;
  handleShadowAction(event, button.dataset.shadowAction, button.dataset.positionId);
});

initDates();
refreshStatus().catch((err) => {
  $("taskMeta").textContent = "状态读取失败";
  $("logs").textContent = String(err);
});
