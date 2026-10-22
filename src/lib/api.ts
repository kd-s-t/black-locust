const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010";

async function handleRes<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body.detail as string) ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export type Dataset = {
  id?: string;
  name: string;
  folder?: string;
  path: string;
  task?: "classify" | "detect" | "unknown";
  has_data_yaml: boolean;
  data_yaml: string | null;
  ready?: boolean;
  original_filename?: string | null;
  created_at?: string;
};

/** Stable id used for select value + training API */
export function datasetKey(d: Dataset): string {
  return d.folder || d.name;
}

export type YoloModel = {
  name: string;
  path: string;
  size_bytes: number;
  modified_at: string;
};

export type Job = {
  id: string;
  dataset_id?: string | null;
  dataset_name: string;
  dataset_folder?: string;
  data_yaml: string;
  base_model: string;
  epochs: number;
  current_epoch?: number;
  imgsz: number;
  status: string;
  message: string;
  model_path: string | null;
  model_name: string | null;
  error: string | null;
  created_at: string;
};

export const api = {
  base: API_BASE,

  listDatasets(): Promise<Dataset[]> {
    return fetch(`${API_BASE}/api/datasets`).then((r) => handleRes(r));
  },

  uploadDataset(file: File, name?: string): Promise<Dataset> {
    const fd = new FormData();
    fd.append("file", file);
    if (name) fd.append("name", name);
    return fetch(`${API_BASE}/api/datasets`, { method: "POST", body: fd }).then((r) =>
      handleRes(r)
    );
  },

  deleteDataset(ref: string): Promise<{ deleted: boolean; id: string; folder: string; name: string }> {
    return fetch(`${API_BASE}/api/datasets/${encodeURIComponent(ref)}`, {
      method: "DELETE",
    }).then((r) => handleRes(r));
  },

  listModels(): Promise<YoloModel[]> {
    return fetch(`${API_BASE}/api/models`).then((r) => handleRes(r));
  },

  deleteModel(name: string): Promise<{ deleted: boolean; name: string }> {
    return fetch(`${API_BASE}/api/models/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }).then((r) => handleRes(r));
  },

  listJobs(): Promise<Job[]> {
    return fetch(`${API_BASE}/api/jobs`).then((r) => handleRes(r));
  },

  startTrain(body: {
    dataset_name: string;
    epochs?: number;
    imgsz?: number;
    base_model?: string;
    model_name?: string;
  }): Promise<Job> {
    return fetch(`${API_BASE}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => handleRes(r));
  },

  predict(file: File, model: string): Promise<PredictResult> {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("model", model);
    return fetch(`${API_BASE}/api/predict`, { method: "POST", body: fd }).then((r) =>
      handleRes(r)
    );
  },
};

export type PredictResult = {
  model: string;
  task: string;
  predictions: Array<{
    label: string;
    confidence: number;
    box?: number[] | null;
  }>;
  image_base64: string | null;
};
