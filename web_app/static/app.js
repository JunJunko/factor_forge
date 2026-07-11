const $ = (id) => document.getElementById(id);

const WORKFLOW_LABELS = [
  ["data", "行情与股票池"],
  ["Timing", "Timing 仓位"],
  ["candidates", "候选股排序"],
  ["execution", "交易计划"],
];

let pollTimer = null;
let latestPayload = null;
let latestSignal = null;
let statusInitialized = false;
let historyItems = [];
let historySelectedDate = null;

function todayShanghai() {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const value = (type) => parts.find((part) => part.type === type).value;
  return `${value("year")}-${value("month")}-${value("day")}`;
}

function ymdCompact(value) {
  return String(value || "").replaceAll("-", "");
}

function shortDate(value) {
  return value ? String(value).slice(0, 10) : "--";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function fmtNum(value, digits = 3) {
  const parsed = number(value);
  return parsed === null ? "--" : parsed.toFixed(digits);
}

function fmtPct(value, digits = 1) {
  const parsed = number(value);
  return parsed === null ? "--" : `${(parsed * 100).toFixed(digits)}%`;
}

function fmtSigned(value, digits = 3) {
  const parsed = number(value);
  if (parsed === null) return "--";
  return `${parsed >= 0 ? "+" : ""}${parsed.toFixed(digits)}`;
}

function fmtCny(value) {
  const parsed = number(value);
  if (parsed === null) return "--";
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 0,
  }).format(parsed);
}

function fmtAmount(value) {
  const parsed = number(value);
  if (parsed === null) return "--";
  if (Math.abs(parsed) >= 1e8) return `${(parsed / 1e8).toFixed(2)}亿`;
  if (Math.abs(parsed) >= 1e4) return `${(parsed / 1e4).toFixed(1)}万`;
  return parsed.toFixed(0);
}

function capital() {
  return Math.max(0, number($("accountCapital").value) || 110000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || response.statusText);
  }
  return response.json();
}

function stateClass(state) {
  const normalized = String(state || "").toUpperCase();
  if (["READY", "HEALTHY", "PASS", "FULL", "RECOVERY"].includes(normalized)) return "ready";
  if (["OBSERVE", "CHECK", "WEAKENING", "MIXED", "WATCH", "REDUCED"].includes(normalized)) return "check";
  if (["FAILED", "BROKEN", "RISK_OFF", "MISSING"].includes(normalized)) return "failed";
  return "";
}

function stateBadgeClass(state) {
  const cls = stateClass(state);
  return cls === "ready" ? "good" : cls === "failed" ? "bad" : cls === "check" ? "warn" : "neutral";
}

function stateLabel(state) {
  const labels = {
    READY: "已就绪",
    HEALTHY: "健康",
    RECOVERY: "恢复中",
    WEAKENING: "转弱",
    BROKEN: "失效",
    MIXED: "分歧",
    OBSERVE: "观察",
    CHECK: "待检查",
    MISSING: "缺失",
    NO_SIGNAL: "未生成",
    RISK_OFF: "风险关闭",
  };
  return labels[String(state || "").toUpperCase()] || String(state || "--");
}

function renderWorkflow(items = []) {
  const mapped = new Map(items.map((item) => [item.name, item]));
  const aliases = {
    data: mapped.get("数据"),
    Timing: mapped.get("Timing"),
    candidates: mapped.get("候选") || mapped.get("信号"),
    execution: mapped.get("执行") || mapped.get("仓位"),
  };
  $("workflow").innerHTML = WORKFLOW_LABELS.map(([key, label]) => {
    const item = aliases[key] || {};
    const state = item.state || "MISSING";
    return `<div class="workflowItem ${stateClass(state)}">
      <div class="workflowTop"><strong>${escapeHtml(label)}</strong><span class="statusDot"></span></div>
      <p>${escapeHtml(item.detail || stateLabel(state))}</p>
    </div>`;
  }).join("");
}

function renderTaskWorkflow(logs, status) {
  const has = (token) => logs.some((line) => line.includes(token));
  const complete = status === "succeeded";
  const failed = status === "failed";
  const states = [
    { name: "数据", state: complete || has("[update_data] done") || has("[update_data] skipped") ? "READY" : has("[update_data]") ? "RUNNING" : "MISSING", detail: "行情与股票池" },
    { name: "Timing", state: complete || has("target_position=") ? "READY" : has("[update_data]") ? "RUNNING" : "MISSING", detail: "仓位模型" },
    { name: "候选", state: complete || has("[generate_all_model_signals] done") ? "READY" : has("[generate_all_model_signals]") ? "RUNNING" : "MISSING", detail: "Alpha + Reliability" },
    { name: "执行", state: complete ? "READY" : failed ? "FAILED" : has("[compute_health]") ? "RUNNING" : "MISSING", detail: "仓位与审计" },
  ];
  renderWorkflow(states);
  document.querySelectorAll(".workflowItem").forEach((item, index) => {
    if (states[index].state === "RUNNING") item.classList.add("running");
  });
}

function renderWarnings(warnings) {
  const element = $("dailyWarnings");
  const items = warnings || [];
  element.classList.toggle("hidden", items.length === 0);
  element.innerHTML = items.map((item) => `<div>${escapeHtml(item)}</div>`).join("");
}

function plannedRows(signal) {
  const account = capital();
  return (signal?.top || []).map((row) => {
    const weight = number(row.target_weight) || 0;
    const close = number(row.raw_close) || 0;
    const targetAmount = account * weight;
    const lotCost = close * 100;
    const shares = lotCost > 0 ? Math.floor(targetAmount / lotCost) * 100 : 0;
    return {
      ...row,
      weight,
      targetAmount,
      shares,
      deployableAmount: shares * close,
      lotBlocked: weight > 0 && shares === 0,
    };
  });
}

function renderDecision(signal) {
  const summary = signal?.summary || {};
  const exposure = number(summary.final_exposure);
  const rows = plannedRows(signal);
  const deployable = rows.reduce((total, row) => total + row.deployableAmount, 0);
  const planned = capital() * (exposure || 0);
  const execution = shortDate(summary.entry_date_for_timing);
  const signalDate = shortDate(summary.signal_date);

  $("timingExposure").textContent = exposure === null ? "--" : fmtPct(exposure, 0);
  $("plannedCapital").textContent = exposure === null ? "--" : fmtCny(planned);
  $("deployableCapital").textContent = exposure === null ? "--" : fmtCny(deployable);
  $("executionDate").textContent = execution;

  if (!signal?.summary) {
    $("decisionHeading").textContent = "等待生成交易计划";
    $("decisionText").textContent = "选择信号日期后运行主链路。";
    return;
  }
  if ((exposure || 0) <= 0) {
    $("decisionHeading").textContent = "今日不新建仓";
    $("decisionText").textContent = `${signalDate} 收盘信号已生成，Timing 仓位为 0%。已有持仓仍按卖出规则检查。`;
    return;
  }
  const blocked = rows.filter((row) => row.lotBlocked).length;
  $("decisionHeading").textContent = `${execution} 开盘，按 ${fmtPct(exposure, 0)} 仓位执行`;
  $("decisionText").textContent = blocked
    ? `模型推荐 ${rows.length} 只股票；按当前资金估算，有 ${blocked} 只不足一手，实际可部署金额低于模型计划。`
    : `模型推荐 ${rows.length} 只股票，整手估算可以覆盖全部候选；开盘前复核停牌与涨跌停状态。`;
}

function renderCandidates(signal) {
  latestSignal = signal || null;
  const rows = plannedRows(signal);
  const tbody = $("topRows");
  const summary = signal?.summary || {};

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="emptyCell">暂无交易计划</td></tr>';
    $("candidateSummary").textContent = "尚未生成候选股。";
    $("allocationWarning").classList.add("hidden");
    $("fileLinks").innerHTML = "";
    renderDecision(signal);
    return;
  }

  tbody.innerHTML = rows.map((row) => {
    const rankChange = number(row.rank_change) || 0;
    const movementClass = rankChange > 0 ? "up" : rankChange < 0 ? "down" : "flat";
    const movement = rankChange > 0 ? `上升 ${rankChange}` : rankChange < 0 ? `下降 ${Math.abs(rankChange)}` : "不变";
    const lotText = row.lotBlocked ? "不足一手" : `${row.shares.toLocaleString("zh-CN")} 股`;
    return `<tr>
      <td class="checkCell"><input class="candidateCheck" type="checkbox" value="${escapeHtml(row.ts_code)}" ${row.weight > 0 ? "checked" : ""} /></td>
      <td><strong>${row.final_rank ?? row.rank ?? "--"}</strong><div class="rankMove ${movementClass}">${movement}</div></td>
      <td><div class="stockName">${escapeHtml(row.name)}</div><div class="stockCode">${escapeHtml(row.ts_code)}</div></td>
      <td>${escapeHtml(row.industry_l1_name || "--")}</td>
      <td>${row.alpha_rank ?? "--"}</td>
      <td><strong>${fmtPct(row.signal_reliability_probability, 1)}</strong></td>
      <td class="scoreBoost">${fmtSigned(row.reliability_adjustment, 4)}</td>
      <td>${fmtPct(row.weight, 1)}</td>
      <td>${fmtCny(row.targetAmount)}</td>
      <td class="${row.lotBlocked ? "lotBlocked" : "tradeReady"}">${lotText}</td>
      <td>¥${fmtNum(row.raw_close, 2)}</td>
    </tr>`;
  }).join("");

  const blocked = rows.filter((row) => row.lotBlocked);
  const deployable = rows.reduce((total, row) => total + row.deployableAmount, 0);
  const warning = $("allocationWarning");
  warning.classList.toggle("hidden", blocked.length === 0);
  warning.textContent = blocked.length
    ? `${blocked.map((row) => row.name).join("、")} 按目标权重不足一手。整手估算可部署 ${fmtCny(deployable)}；未自动把剩余资金转配给其他股票。`
    : "";

  $("candidateSummary").textContent = `${shortDate(summary.signal_date)} 收盘后生成，计划在 ${shortDate(summary.entry_date_for_timing)} 开盘执行。`;
  const fileLabels = {
    summary: "信号摘要",
    top_recommendations: "Top5 CSV",
    top100_candidates: "Top100 CSV",
    report: "信号报告",
    run_log: "运行日志",
  };
  $("fileLinks").innerHTML = Object.entries(signal.files || {})
    .map(([name, path]) => `<a href="/api/file?path=${encodeURIComponent(path)}" target="_blank">${escapeHtml(fileLabels[name] || name)}</a>`)
    .join("");
  renderDecision(signal);
}

function renderHealth(monitoring) {
  const health = monitoring?.factor_health;
  if (!health) {
    $("healthState").textContent = "无数据";
    $("healthState").className = "stateBadge neutral";
    $("healthFreshness").textContent = "尚未找到 factor_health_daily。";
    $("healthMetrics").innerHTML = "";
    return;
  }
  $("healthState").textContent = stateLabel(health.state);
  $("healthState").className = `stateBadge ${stateBadgeClass(health.state)}`;
  $("healthFreshness").textContent = health.is_stale
    ? `健康度截至 ${health.date}，滞后 ${health.lag_calendar_days} 个自然日。该指标依赖已兑现收益，仅作监控，不直接改变今日仓位。`
    : `健康度截至 ${health.date}，当前状态用于监控因子排序质量。`;
  const metrics = [
    ["20日 RankIC", fmtNum(health.rolling_rank_ic_20, 3)],
    ["60日 RankIC", fmtNum(health.rolling_rank_ic_60, 3)],
    ["20日 ICIR", fmtNum(health.icir_20, 2)],
    ["20日 Spread", fmtPct(health.spread_20, 2)],
    ["60日 Spread", fmtPct(health.spread_60, 2)],
    ["分层单调性", fmtNum(health.decile_monotonicity, 2)],
  ];
  $("healthMetrics").innerHTML = metrics
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`)
    .join("");
}

function renderReliability(monitoring, signal) {
  const live = monitoring?.live_stock_reliability || {};
  const impact = live.impact || signal?.summary?.reliability_impact || {};
  const rows = plannedRows(signal);
  $("reliabilityMean").textContent = live.probability_mean === null || live.probability_mean === undefined
    ? "--"
    : `均值 ${fmtPct(live.probability_mean, 1)}`;
  $("reliabilityMean").className = `stateBadge ${number(live.probability_mean) >= 0.65 ? "good" : "warn"}`;
  if (!rows.length) {
    $("reliabilitySummary").textContent = "等待最新股票可靠性结果。";
    $("reliabilityRows").innerHTML = "";
    return;
  }
  const replaced = impact.top5_replaced_count ?? "--";
  $("reliabilitySummary").textContent = `冻结公式 alpha_score + 0.05 × reliability_zscore；相对纯 Alpha，Top5 替换 ${replaced} 只。Reliability 只调整排序，Timing 单独控制仓位。`;
  $("reliabilityRows").innerHTML = rows.map((row) => {
    const change = number(row.rank_change) || 0;
    const cls = change > 0 ? "up" : change < 0 ? "down" : "flat";
    const movement = change > 0 ? `↑ ${change}` : change < 0 ? `↓ ${Math.abs(change)}` : "--";
    return `<div class="rankImpactRow">
      <strong>${escapeHtml(row.name)} <span class="stockCode">${escapeHtml(row.ts_code)}</span></strong>
      <span class="probability">${fmtPct(row.signal_reliability_probability, 1)}</span>
      <span class="movement rankMove ${cls}">${row.alpha_rank ?? "--"} → ${row.final_rank ?? row.rank ?? "--"} ${movement}</span>
    </div>`;
  }).join("");
}

function renderDetails(payload) {
  const data = payload.data || {};
  const timing = data.timing_position || {};
  const summary = payload.latest_signal?.summary || {};
  const detailRows = [
    ["数据版本", data.version || "--"],
    ["面板范围", `${shortDate(data.start_date)} 至 ${shortDate(data.end_date)}`],
    ["可用交易日", shortDate(data.latest_usable_date)],
    ["数据行数", number(data.row_count)?.toLocaleString("zh-CN") || "--"],
    ["Timing 日期", shortDate(timing.latest_date)],
    ["Timing 仓位", fmtPct(timing.target_position, 0)],
  ];
  $("dataStatus").innerHTML = detailRows.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join("");

  const signalRows = [
    ["信号日期", shortDate(summary.signal_date)],
    ["执行日期", shortDate(summary.entry_date_for_timing)],
    ["模型", summary.model || "--"],
    ["算法版本", summary.signal_algorithm || "--"],
    ["Selector", summary.selector || "--"],
    ["候选样本", summary.predictable_candidates ?? "--"],
  ];
  $("signalStatus").innerHTML = signalRows.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join("");

  $("auditLinks").innerHTML = (payload.dashboard?.research_audit?.files || [])
    .map((item) => `<a href="/api/file?path=${encodeURIComponent(item.path)}" target="_blank">${escapeHtml(item.label)}</a>`)
    .join("");
}

function renderHistoryList(items) {
  historyItems = items || [];
  $("historySummary").textContent = historyItems.length
    ? `已归档 ${historyItems.length} 个信号日；同一日期只显示最新一次冻结策略运行。`
    : "当前冻结策略尚未生成可浏览的历史信号。";
  $("historyRows").innerHTML = historyItems.length
    ? historyItems.map((item) => `<tr>
        <td><strong>${shortDate(item.signal_date)}</strong><div class="stockCode">${item.predictable_candidates ?? "--"} 个候选</div></td>
        <td>${shortDate(item.entry_date)}</td>
        <td>${fmtPct(item.final_exposure, 0)}</td>
        <td>${fmtPct(item.reliability_probability_mean, 1)}</td>
        <td><button class="historyViewButton ${item.signal_date === historySelectedDate ? "active" : ""}" data-history-date="${escapeHtml(item.signal_date)}">查看</button></td>
      </tr>`).join("")
    : '<tr><td colspan="5" class="emptyCell">暂无历史记录</td></tr>';
}

function renderHistoryDetail(signal, item) {
  const summary = signal?.summary || {};
  const rows = signal?.top || [];
  if (!signal?.summary) {
    $("historyDetailTitle").textContent = "选择一个信号日";
    $("historyRunStamp").textContent = "";
    $("historyMetrics").innerHTML = "";
    $("historyCandidateRows").innerHTML = '<tr><td colspan="5" class="emptyCell">选择左侧记录查看当日 Top5</td></tr>';
    return;
  }
  $("historyDetailTitle").textContent = `${shortDate(summary.signal_date)} 的推荐快照`;
  $("historyRunStamp").textContent = item?.generated_at ? `生成于 ${item.generated_at.replace("T", " ")}` : "";
  const metrics = [
    ["执行日", shortDate(summary.entry_date_for_timing)],
    ["Timing 仓位", fmtPct(summary.final_exposure, 0)],
    ["Reliability 均值", fmtPct(summary.reliability_probability_mean, 1)],
    ["可预测候选", summary.predictable_candidates ?? "--"],
  ];
  $("historyMetrics").innerHTML = metrics
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`)
    .join("");
  $("historyCandidateRows").innerHTML = rows.length
    ? rows.map((row) => `<tr>
        <td>${row.final_rank ?? row.rank ?? "--"}</td>
        <td><strong>${escapeHtml(row.name || row.ts_code)}</strong><div class="stockCode">${escapeHtml(row.ts_code)}</div></td>
        <td>${row.alpha_rank ?? "--"}</td>
        <td>${fmtPct(row.signal_reliability_probability, 1)}</td>
        <td>${fmtPct(row.target_weight, 1)}</td>
      </tr>`).join("")
    : '<tr><td colspan="5" class="emptyCell">该日没有候选股</td></tr>';
}

async function loadHistory(signalDate) {
  const date = shortDate(signalDate);
  if (date === "--") return;
  historySelectedDate = date;
  const signal = await api(`/api/signal-history/${encodeURIComponent(date)}`);
  const item = historyItems.find((entry) => entry.signal_date === date);
  renderHistoryList(historyItems);
  renderHistoryDetail(signal, item);
}

async function refreshHistory() {
  const payload = await api("/api/signal-history?limit=120");
  const items = payload.items || [];
  renderHistoryList(items);
  const target = items.some((item) => item.signal_date === historySelectedDate)
    ? historySelectedDate
    : items[0]?.signal_date;
  if (target) await loadHistory(target);
  else renderHistoryDetail(null, null);
}

function renderStatus(payload) {
  latestPayload = payload;
  const signal = payload.latest_signal;
  renderWorkflow(payload.dashboard?.workflow || []);
  renderWarnings(payload.dashboard?.warnings || []);
  renderCandidates(signal);
  renderHealth(payload.monitoring);
  renderReliability(payload.monitoring, signal);
  renderDetails(payload);
  $("lastRefresh").textContent = `刷新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
}

function renderSellAdvice(data) {
  const items = data?.items || [];
  const banner = $("sellBanner");
  banner.classList.toggle("hidden", items.length === 0);
  banner.textContent = items.length ? `已检查 ${items.length} 只持仓，卖出建议 ${items.filter((item) => item.action === "SELL").length} 只。` : "";
  $("sellRows").innerHTML = items.length
    ? items.map((row) => `<tr>
        <td><strong>${escapeHtml(row.name || row.ts_code)}</strong><div class="stockCode">${escapeHtml(row.ts_code)}</div></td>
        <td>${row.holding_trade_days ?? "--"}</td>
        <td><span class="actionBadge ${row.action === "SELL" ? "sell" : "hold"}">${escapeHtml(row.action)}</span></td>
        <td>${escapeHtml(row.reason || "--")}</td>
      </tr>`).join("")
    : '<tr><td colspan="4" class="emptyCell">暂无卖出检查</td></tr>';
}

function renderShadowPortfolio(data) {
  const summary = data?.summary || {};
  const rows = data?.positions || [];
  $("shadowSummary").textContent = rows.length
    ? `开放持仓 ${summary.position_count ?? 0} 只，卖出建议 ${summary.sell_count ?? 0} 只，平均收益 ${fmtPct(summary.avg_shadow_return, 2)}。`
    : "尚未加入影子持仓。";
  $("shadowRows").innerHTML = rows.length
    ? rows.map((row) => {
        const returnValue = number(row.shadow_return);
        const closed = String(row.status || "OPEN") === "CLOSED";
        return `<tr>
          <td><strong>${escapeHtml(row.name || row.ts_code)}</strong><div class="stockCode">${escapeHtml(row.ts_code)}</div></td>
          <td>${row.holding_trade_days ?? "--"}</td>
          <td class="${returnValue > 0 ? "positive" : returnValue < 0 ? "negative" : ""}">${fmtPct(returnValue, 2)}</td>
          <td><span class="actionBadge ${row.sell_action === "SELL" ? "sell" : "hold"}">${escapeHtml(row.sell_action || "HOLD")}</span></td>
          <td>${closed ? "已卖出" : `<button class="miniButton" data-shadow-action="sell" data-position-id="${escapeHtml(row.id)}">卖出</button>`} <button class="miniButton" data-shadow-action="delete" data-position-id="${escapeHtml(row.id)}">删除</button></td>
        </tr>`;
      }).join("")
    : '<tr><td colspan="5" class="emptyCell">暂无影子持仓</td></tr>';
}

function selectedCandidateCodes() {
  return Array.from(document.querySelectorAll(".candidateCheck:checked")).map((input) => input.value).filter(Boolean);
}

function setBusy(busy) {
  ["dailyRunBtn", "syncBtn", "signalBtn", "sellAdviceBtn", "addShadowBtn", "refreshShadowBtn", "refreshBtn", "refreshHistoryBtn"].forEach((id) => {
    if ($(id)) $(id).disabled = busy;
  });
  $("runStatus").textContent = busy ? "运行中" : "就绪";
}

async function refreshShadowPortfolio() {
  renderShadowPortfolio(await api("/api/shadow-portfolio"));
}

async function refreshStatus() {
  const payload = await api("/api/status");
  const defaultDate = shortDate(payload.default_signal_date);
  if (!statusInitialized && defaultDate !== "--") {
    $("signalDate").value = defaultDate;
    $("syncStart").value = ymdCompact(defaultDate);
    $("syncEnd").value = ymdCompact(defaultDate);
    statusInitialized = true;
  }
  renderStatus(payload);
  await Promise.all([
    refreshShadowPortfolio().catch(() => {}),
    refreshHistory().catch(() => {}),
  ]);
}

function startPolling(taskId) {
  clearInterval(pollTimer);
  document.querySelector(".advancedSection").open = true;
  pollTask(taskId);
  pollTimer = setInterval(() => pollTask(taskId), 1200);
}

async function pollTask(taskId) {
  const task = await api(`/api/tasks/${taskId}`);
  const logs = task.logs || [];
  $("taskMeta").textContent = `任务 ${task.id} · ${task.status}`;
  $("logs").textContent = logs.join("\n");
  $("logs").scrollTop = $("logs").scrollHeight;
  renderTaskWorkflow(logs, task.status);
  if (["succeeded", "failed"].includes(task.status)) {
    clearInterval(pollTimer);
    pollTimer = null;
    setBusy(false);
    $("runStatus").textContent = task.status === "succeeded" ? "已完成" : "运行失败";
    if (task.status === "succeeded") await refreshStatus();
    if (task.status === "failed") throw new Error(task.error || "任务执行失败");
  }
}

async function runDailyChain() {
  const signalDate = $("signalDate").value;
  if (!signalDate) throw new Error("请选择信号日期。");
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
  const result = await api("/api/daily-chain", { method: "POST", body: JSON.stringify(payload) });
  startPolling(result.task_id);
}

async function runSync() {
  const payload = {
    start: $("syncStart").value.trim() || ymdCompact($("signalDate").value),
    end: $("syncEnd").value.trim() || ymdCompact($("signalDate").value),
    merge_full_history: $("mergeFull").checked,
  };
  setBusy(true);
  const result = await api("/api/sync", { method: "POST", body: JSON.stringify(payload) });
  startPolling(result.task_id);
}

async function runSignal() {
  setBusy(true);
  const result = await api("/api/signal", { method: "POST", body: JSON.stringify({ signal_date: $("signalDate").value }) });
  startPolling(result.task_id);
}

async function runSellAdvice() {
  const holdings = $("holdingsText").value;
  localStorage.setItem("factorForgeHoldings", holdings);
  renderSellAdvice(await api("/api/sell-advice", {
    method: "POST",
    body: JSON.stringify({ signal_date: $("signalDate").value, holdings_text: holdings }),
  }));
  document.querySelector(".operationsSection").open = true;
}

async function addSelectedToShadow() {
  const codes = selectedCandidateCodes();
  if (!codes.length) throw new Error("请先勾选候选股。");
  const signalDate = shortDate(latestSignal?.summary?.signal_date || $("signalDate").value);
  const result = await api("/api/shadow-portfolio/add", {
    method: "POST",
    body: JSON.stringify({ signal_date: signalDate, ts_codes: codes }),
  });
  renderShadowPortfolio(result.portfolio);
  document.querySelector(".operationsSection").open = true;
}

async function updateShadowPosition(positionId, action) {
  if (!positionId) return;
  if (action === "delete" && !window.confirm("删除这条影子持仓？")) return;
  const result = await api(`/api/shadow-portfolio/${action}`, {
    method: "POST",
    body: JSON.stringify({ position_id: positionId }),
  });
  renderShadowPortfolio(result.portfolio);
}

function showError(error) {
  setBusy(false);
  $("runStatus").textContent = "操作失败";
  $("taskMeta").textContent = "操作失败";
  $("logs").textContent = String(error?.message || error);
  document.querySelector(".advancedSection").open = true;
}

function syncDateInputs() {
  const compact = ymdCompact($("signalDate").value);
  $("syncStart").value = compact;
  $("syncEnd").value = compact;
}

function init() {
  const today = todayShanghai();
  $("signalDate").value = today;
  $("syncStart").value = ymdCompact(today);
  $("syncEnd").value = ymdCompact(today);
  $("accountCapital").value = localStorage.getItem("factorForgeCapital") || "110000";
  $("holdingsText").value = localStorage.getItem("factorForgeHoldings") || "";
  renderWorkflow([]);
}

$("refreshBtn").addEventListener("click", () => refreshStatus().catch(showError));
$("dailyRunBtn").addEventListener("click", () => runDailyChain().catch(showError));
$("syncBtn").addEventListener("click", () => runSync().catch(showError));
$("signalBtn").addEventListener("click", () => runSignal().catch(showError));
$("sellAdviceBtn").addEventListener("click", () => runSellAdvice().catch(showError));
$("addShadowBtn").addEventListener("click", () => addSelectedToShadow().catch(showError));
$("refreshShadowBtn").addEventListener("click", () => refreshShadowPortfolio().catch(showError));
$("refreshHistoryBtn").addEventListener("click", () => refreshHistory().catch(showError));
$("signalDate").addEventListener("change", syncDateInputs);
$("accountCapital").addEventListener("input", () => {
  localStorage.setItem("factorForgeCapital", $("accountCapital").value);
  renderCandidates(latestSignal);
});
$("shadowRows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-shadow-action]");
  if (!button) return;
  updateShadowPosition(button.dataset.positionId, button.dataset.shadowAction).catch(showError);
});
$("historyRows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-history-date]");
  if (!button) return;
  loadHistory(button.dataset.historyDate).catch(showError);
});

init();
refreshStatus().catch(showError);
