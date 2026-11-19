async function init() {
  Object.assign(els, {
    shoeCode: $("shoeCode"),
    angleTag: $("angleTag"),
    dateStamp: $("dateStamp"),
    todayBtn: $("todayBtn"),
    tagList: $("tagList"),
    manageTagsBtn: $("manageTagsBtn"),
    tagManager: $("tagManager"),
    newTagInput: $("newTagInput"),
    addTagBtn: $("addTagBtn"),
    masterDropZone: $("masterDropZone"),
    nextSlotHint: $("nextSlotHint"),
    clearQueueBtn: $("clearQueueBtn"),
    queueCount: $("queueCount"),
    filenamePreview: $("filenamePreview"),
    previewNote: $("previewNote"),
    preferOriginal: $("preferOriginal"),
    addDate: $("addDate"),
    clearImagesAfter: $("clearImagesAfter"),
    clearAngleAfter: $("clearAngleAfter"),
    clearShoeAfter: $("clearShoeAfter"),
    directoryName: $("directoryName"),
    directoryStatus: $("directoryStatus"),
    chooseDirectoryBtn: $("chooseDirectoryBtn"),
    downloadBtn: $("downloadBtn"),
    toast: $("toast")
  });

  const stored = await chrome.storage.local.get({
    [QUEUE_KEY]: Array.from({ length: QUEUE_SIZE }, () => null),
    [FORM_KEY]: { xxx: "", y: "", date: todayString() },
    [SETTINGS_KEY]: DEFAULT_SETTINGS,
    [TAGS_KEY]: DEFAULT_TAGS
  });

  queue = normalizeQueue(stored[QUEUE_KEY]);
  settings = { ...DEFAULT_SETTINGS, ...(stored[SETTINGS_KEY] || {}) };
  tags = Array.isArray(stored[TAGS_KEY]) && stored[TAGS_KEY].length ? stored[TAGS_KEY] : [...DEFAULT_TAGS];

  const form = stored[FORM_KEY] || {};
  els.shoeCode.value = form.xxx || "";
  els.angleTag.value = form.y || "";
  els.dateStamp.value = /^\d{8}$/.test(form.date || "") ? form.date : todayString();

  els.preferOriginal.checked = settings.preferOriginal;
  els.addDate.checked = settings.addDate;
  els.clearImagesAfter.checked = settings.clearImagesAfter;
  els.clearAngleAfter.checked = settings.clearAngleAfter;
  els.clearShoeAfter.checked = settings.clearShoeAfter;

  try {
    directoryHandle = await loadDirectoryHandle();
    directoryPermission = directoryHandle ? "unknown" : "missing";
  } catch (error) {
    console.warn("Unable to restore download directory", error);
    directoryHandle = null;
    directoryPermission = "missing";
  }

  [els.shoeCode, els.angleTag, els.dateStamp].forEach((input) => {
    input.addEventListener("input", () => {
      persistFormSoon();
      renderAll();
    });
  });

  els.dateStamp.addEventListener("blur", () => {
    const digits = els.dateStamp.value.replace(/\D/g, "").slice(0, 8);
    els.dateStamp.value = digits.length === 8 ? digits : todayString();
    persistFormSoon();
    renderAll();
  });

  els.todayBtn.addEventListener("click", () => {
    els.dateStamp.value = todayString();
    persistFormSoon();
    renderAll();
  });

  els.manageTagsBtn.addEventListener("click", () => {
    tagManagerOpen = !tagManagerOpen;
    els.tagManager.classList.toggle("is-hidden", !tagManagerOpen);
    els.manageTagsBtn.textContent = tagManagerOpen ? "完成" : "管理标签";
    renderTags();
  });

  const addTag = async () => {
    const value = els.newTagInput.value.trim();
    if (!value) return;
    if (!tags.includes(value)) {
      tags.push(value);
      await persistTags();
    }
    els.newTagInput.value = "";
    renderTags();
  };
  els.addTagBtn.addEventListener("click", addTag);
  els.newTagInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") addTag();
  });

  bindDropTarget(els.masterDropZone, null);
  bindSlotReordering();

  els.clearQueueBtn.addEventListener("click", async () => {
    queue = Array.from({ length: QUEUE_SIZE }, () => null);
    await persistQueue();
    await chrome.runtime.sendMessage({ type: "CLEAR_DRAG_STATE" }).catch(() => {});
    renderAll();
  });

  [els.preferOriginal, els.addDate, els.clearImagesAfter, els.clearAngleAfter, els.clearShoeAfter]
    .forEach((input) => input.addEventListener("change", persistSettings));
  els.chooseDirectoryBtn.addEventListener("click", chooseDirectory);
  els.downloadBtn.addEventListener("click", startDownload);

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes[QUEUE_KEY]) {
      queue = normalizeQueue(changes[QUEUE_KEY].newValue);
      renderAll();
      hydrateQueuePreviews().catch(() => {});
    }
    if (area === "session" && changes[NOTICE_KEY]?.newValue) {
      const notice = changes[NOTICE_KEY].newValue;
      if (Date.now() - (notice.at || 0) < 5000) showToast(notice.message, notice.type);
    }
  });

  const notice = (await chrome.storage.session.get(NOTICE_KEY))[NOTICE_KEY];
  if (notice && Date.now() - (notice.at || 0) < 3000) {
    showToast(notice.message, notice.type);
  }

  renderAll();
  hydrateQueuePreviews().catch(() => {});
}

init().catch((error) => {
  console.error(error);
});
