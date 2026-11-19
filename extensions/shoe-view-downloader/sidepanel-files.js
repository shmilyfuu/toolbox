function openHandleDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(HANDLE_DB_NAME, HANDLE_DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(HANDLE_STORE_NAME)) {
        db.createObjectStore(HANDLE_STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("无法打开文件夹记录数据库。"));
  });
}

async function saveDirectoryHandle(handle) {
  const db = await openHandleDb();
  try {
    await new Promise((resolve, reject) => {
      const tx = db.transaction(HANDLE_STORE_NAME, "readwrite");
      tx.objectStore(HANDLE_STORE_NAME).put(handle, DIRECTORY_HANDLE_KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error || new Error("保存文件夹记录失败。"));
      tx.onabort = () => reject(tx.error || new Error("保存文件夹记录失败。"));
    });
  } finally {
    db.close();
  }
}

async function loadDirectoryHandle() {
  const db = await openHandleDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(HANDLE_STORE_NAME, "readonly");
      const request = tx.objectStore(HANDLE_STORE_NAME).get(DIRECTORY_HANDLE_KEY);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error("读取文件夹记录失败。"));
    });
  } finally {
    db.close();
  }
}

async function queryDirectoryPermission() {
  if (!directoryHandle) {
    directoryPermission = "missing";
    return directoryPermission;
  }
  if (typeof directoryHandle.queryPermission !== "function") {
    directoryPermission = "unknown";
    return directoryPermission;
  }
  try {
    directoryPermission = await directoryHandle.queryPermission({ mode: "readwrite" });
  } catch {
    directoryPermission = "prompt";
  }
  return directoryPermission;
}

async function chooseDirectory() {
  if (!("showDirectoryPicker" in window)) {
    showToast("当前浏览器不支持指定文件夹写入，请使用较新的 Chrome 或 Edge。", "error");
    return false;
  }

  try {
    const options = { mode: "readwrite", id: "shoe-view-downloader" };
    options.startIn = directoryHandle || "downloads";
    const picked = await window.showDirectoryPicker(options);
    directoryHandle = picked;
    await saveDirectoryHandle(picked);
    directoryPermission = "unknown";
    renderAll();
    showToast(`保存位置已设置为「${picked.name}」。`, "success");
    return true;
  } catch (error) {
    if (error?.name === "AbortError") return false;
    showToast(error?.message || "选择文件夹失败。", "error");
    return false;
  }
}

async function ensureDirectoryPermission() {
  if (!directoryHandle) return false;
  if (directoryPermission === "granted") return true;
  if (typeof directoryHandle.requestPermission !== "function") return true;

  try {
    directoryPermission = await directoryHandle.requestPermission({ mode: "readwrite" });
    renderAll();
    return directoryPermission === "granted";
  } catch (error) {
    directoryPermission = "denied";
    renderAll();
    throw error;
  }
}

function extensionFromBlob(blob, fallback = "jpg") {
  const clean = String(blob?.type || "").split(";")[0].trim().toLowerCase();
  const map = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/avif": "avif",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tif",
    "image/svg+xml": "svg"
  };
  return map[clean] || String(fallback || "jpg").replace(/^\./, "").toLowerCase();
}

async function filenameExists(dirHandle, filename) {
  try {
    await dirHandle.getFileHandle(filename);
    return true;
  } catch (error) {
    if (error?.name === "NotFoundError") return false;
    throw error;
  }
}

async function uniqueFilename(dirHandle, filename) {
  if (!(await filenameExists(dirHandle, filename))) return filename;
  const dot = filename.lastIndexOf(".");
  const stem = dot > 0 ? filename.slice(0, dot) : filename;
  const ext = dot > 0 ? filename.slice(dot) : "";
  for (let n = 1; n <= 9999; n += 1) {
    const candidate = `${stem} (${n})${ext}`;
    if (!(await filenameExists(dirHandle, candidate))) return candidate;
  }
  throw new Error(`文件名冲突过多：${filename}`);
}


function loadedResourceToBlob(result) {
  if (!result?.found || typeof result.content !== "string") return null;
  const mime = result.mimeType || "application/octet-stream";
  if (result.base64Encoded) return base64ToBlob(result.content, mime);
  return new Blob([result.content], { type: mime });
}

async function tryLoadedImageBlob(entry) {
  if (!entry?.url || !/^https?:/i.test(entry.url)) return null;
  try {
    const result = await chrome.runtime.sendMessage({
      type: "READ_LOADED_RESOURCE",
      tabId: Number.isInteger(entry.tabId) ? entry.tabId : null,
      url: entry.url
    });
    if (!result?.ok || !result.found) return null;
    const blob = loadedResourceToBlob(result);
    const imageLike = result.resourceType === "Image" || String(blob?.type || result.mimeType || "").startsWith("image/");
    return blob?.size && imageLike ? blob : null;
  } catch {
    return null;
  }
}

async function fetchImageBlob(url) {
  const response = await fetch(url, {
    method: "GET",
    credentials: "include",
    cache: "default",
    redirect: "follow"
  });
  if (!response.ok) throw new Error(`图片请求失败（HTTP ${response.status}）。`);
  const blob = await response.blob();
  if (!blob.size) throw new Error("图片内容为空。");
  return blob;
}

async function savePreparedItem(entry) {
  if (!entry?.ok) return entry;
  try {
    const blob = await tryLoadedImageBlob(entry) || await fetchImageBlob(entry.url);
    const ext = extensionFromBlob(blob, entry.extGuess);
    const desiredName = `${entry.baseName}.${ext}`;
    const filename = await uniqueFilename(directoryHandle, desiredName);
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    try {
      await writable.write(blob);
      await writable.close();
    } catch (error) {
      try {
        if (typeof writable.abort === "function") await writable.abort();
      } catch {
        // Ignore cleanup failures and report the original write error.
      }
      throw error;
    }
    return { index: entry.index, ok: true, filename };
  } catch (error) {
    return { index: entry.index, ok: false, error: error?.message || String(error) };
  }
}

async function startDownload() {
  const count = queue.filter(Boolean).length;
  if (!count) return;
  if (!els.shoeCode.value.trim()) {
    showToast("请填写鞋款 xxx。", "warning");
    els.shoeCode.focus();
    return;
  }
  if (!els.angleTag.value.trim()) {
    showToast("请填写角度 y。", "warning");
    els.angleTag.focus();
    return;
  }
  if (!directoryHandle) {
    showToast("请先选择保存文件夹。", "warning");
    return;
  }

  try {
    const permitted = await ensureDirectoryPermission();
    if (!permitted) {
      showToast("需要获得该文件夹的写入权限。", "warning");
      return;
    }
  } catch {
    showToast("文件夹写入权限已失效，请重新选择文件夹。", "error");
    return;
  }

  els.downloadBtn.disabled = true;
  els.downloadBtn.classList.add("busy");
  els.downloadBtn.textContent = "正在保存…";

  try {
    const response = await chrome.runtime.sendMessage({
      type: "PREPARE_BATCH",
      payload: {
        items: queue,
        xxx: els.shoeCode.value.trim(),
        y: els.angleTag.value.trim(),
        date: els.dateStamp.value.trim() || todayString(),
        addDate: settings.addDate,
        preferOriginal: settings.preferOriginal
      }
    });

    if (!response?.ok) throw new Error(response?.error || "图片准备失败");

    const prepared = Array.isArray(response.results) ? response.results : [];
    const preparationFailures = prepared.filter((entry) => !entry.ok);
    const writableItems = prepared.filter((entry) => entry.ok);
    const saved = await Promise.all(writableItems.map(savePreparedItem));
    const allResults = [...preparationFailures, ...saved].sort((a, b) => a.index - b.index);
    const succeeded = allResults.filter((entry) => entry.ok).length;
    const failed = allResults.length - succeeded;

    if (failed) {
      const firstError = allResults.find((item) => !item.ok)?.error || "部分图片保存失败";
      showToast(`成功 ${succeeded} 张，失败 ${failed} 张：${firstError}`, "warning");
    } else {
      showToast(`已保存 ${succeeded} 张图片到「${directoryHandle.name}」。`, "success");
    }

    if (succeeded > 0 && failed === 0) {
      if (settings.clearImagesAfter) {
        queue = Array.from({ length: QUEUE_SIZE }, () => null);
        await persistQueue();
        await chrome.runtime.sendMessage({ type: "CLEAR_DRAG_STATE" }).catch(() => {});
      }
      if (settings.clearAngleAfter) els.angleTag.value = "";
      if (settings.clearShoeAfter) els.shoeCode.value = "";
      persistFormSoon();
    }
  } catch (error) {
    showToast(error?.message || String(error), "error");
  } finally {
    els.downloadBtn.classList.remove("busy");
    renderAll();
  }
}

