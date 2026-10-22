"use client";

import { useCallback, useEffect, useState, type DragEvent } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Loader2, Play, X } from "lucide-react";
import {
  api,
  datasetKey,
  type Dataset,
  type Job,
  type PredictResult,
  type YoloModel,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

type Step = "dataset" | "train" | "test";

const stepMotion = {
  initial: { opacity: 0, y: 18 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] as const },
  },
  exit: { opacity: 0, y: -10, transition: { duration: 0.2 } },
};

function parseStep(raw: string | null): Step {
  if (raw === "train" || raw === "test") return raw;
  return "dataset";
}

function slugPart(value: string, max = 40): string {
  return value
    .replace(/\.pt$/i, "")
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")
    .slice(0, max);
}

/** Nice unique default: cats_dogs_yolo26n-cls, then -v2, -v3… */
function suggestModelName(dataset: string, base: string, existing: string[]): string {
  const ds = slugPart(dataset) || "model";
  const bm = slugPart(base) || "yolo";
  const root = `${ds}_${bm}`;
  const taken = new Set(
    existing.map((n) => n.replace(/\.pt$/i, "").toLowerCase())
  );
  if (!taken.has(root.toLowerCase())) return root;
  let n = 2;
  while (taken.has(`${root}-v${n}`.toLowerCase())) n += 1;
  return `${root}-v${n}`;
}

export default function YoloTrain() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const step = parseStep(searchParams.get("step"));
  const selectedDataset = searchParams.get("dataset") ?? "";
  const baseModel = searchParams.get("model") ?? "";
  const epochs = Math.min(500, Math.max(1, Number(searchParams.get("epochs")) || 50));
  const modelName = searchParams.get("name") ?? "";
  const uploadOpen = searchParams.get("upload") === "1";
  const testModel = searchParams.get("testModel") ?? "";

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<YoloModel[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [datasetName, setDatasetName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [training, setTraining] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [predictDrag, setPredictDrag] = useState(false);
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [nameTouched, setNameTouched] = useState(false);

  const setParams = useCallback(
    (updates: Record<string, string | null>, replace = true) => {
      const params = new URLSearchParams(searchParams.toString());
      for (const [key, value] of Object.entries(updates)) {
        if (value == null || value === "") params.delete(key);
        else params.set(key, value);
      }
      const qs = params.toString();
      const href = qs ? `${pathname}?${qs}` : pathname;
      if (replace) router.replace(href, { scroll: false });
      else router.push(href, { scroll: false });
    },
    [pathname, router, searchParams]
  );

  const openUpload = () => setParams({ upload: "1", step: "dataset" });
  const closeUpload = () => {
    setDragOver(false);
    setParams({ upload: null });
  };

  const refresh = useCallback(async () => {
    try {
      const [ds, ms, js] = await Promise.all([
        api.listDatasets(),
        api.listModels(),
        api.listJobs(),
      ]);
      setDatasets(ds);
      setModels(ms);
      setJobs(js);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!uploadOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !uploading) setParams({ upload: null });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [uploadOpen, uploading, setParams]);

  useEffect(() => {
    if (step !== "train") return;
    if (!selectedDataset) {
      setParams({ step: "dataset" });
      return;
    }
    if (datasets.length === 0) return;
    const ok = datasets.some(
      (d) => datasetKey(d) === selectedDataset || d.id === selectedDataset || d.name === selectedDataset
    );
    if (!ok) setParams({ step: "dataset", dataset: null });
  }, [step, selectedDataset, datasets, setParams]);

  useEffect(() => {
    if (step !== "test") return;
    if (testModel && models.some((m) => m.name === testModel)) return;
    if (models[0]) setParams({ testModel: models[0].name });
  }, [step, testModel, models, setParams]);

  useEffect(() => {
    if (step !== "train") return;
    if (nameTouched) return;
    if (!selectedDataset || !baseModel) return;
    const ds =
      datasets.find(
        (d) =>
          datasetKey(d) === selectedDataset ||
          d.id === selectedDataset ||
          d.name === selectedDataset
      )?.name || selectedDataset;
    const taken = [
      ...models.map((m) => m.name),
      ...jobs.map((j) => j.model_name).filter((n): n is string => !!n),
    ];
    const suggested = suggestModelName(ds, baseModel, taken);
    if (modelName !== suggested) setParams({ name: suggested });
  }, [
    step,
    nameTouched,
    selectedDataset,
    baseModel,
    datasets,
    models,
    jobs,
    modelName,
    setParams,
  ]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    const active = jobs.some((j) => j.status === "queued" || j.status === "running");
    if (!active) return;
    const id = setInterval(() => {
      api.listJobs().then(setJobs).catch(() => {});
      api.listModels().then(setModels).catch(() => {});
    }, 3000);
    return () => clearInterval(id);
  }, [jobs]);

  const onUpload = async (file: File | null) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setError("Upload a .zip file");
      return;
    }
    setUploading(true);
    setError(null);
    setInfo(null);
    try {
      const result = await api.uploadDataset(file, datasetName || undefined);
      const key = datasetKey(result);
      setInfo(`Uploaded "${result.name}"`);
      setDatasetName("");
      setParams({ dataset: key, step: "dataset", upload: null });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (uploading) return;
    onUpload(e.dataTransfer.files?.[0] ?? null);
  };

  const onDeleteDataset = async (d: Dataset) => {
    const key = datasetKey(d);
    const ref = d.id || key;
    if (!ref) return;
    setError(null);
    setInfo(null);
    try {
      await api.deleteDataset(ref);
      if (selectedDataset === key) setParams({ dataset: null });
      setInfo(`Deleted "${d.name}"`);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const onDeleteModel = async (name: string) => {
    setError(null);
    setInfo(null);
    try {
      await api.deleteModel(name);
      const updates: Record<string, string | null> = {};
      if (baseModel === name) updates.model = null;
      if (testModel === name) updates.testModel = null;
      if (Object.keys(updates).length) setParams(updates);
      setInfo(`Deleted "${name}"`);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const selectedMeta = datasets.find(
    (d) => datasetKey(d) === selectedDataset || d.id === selectedDataset || d.name === selectedDataset
  );
  const running = jobs.some((j) => j.status === "queued" || j.status === "running");
  const takenModelNames = [
    ...models.map((m) => m.name),
    ...jobs.map((j) => j.model_name).filter((n): n is string => !!n),
  ];
  const normalizedModelName = slugPart(modelName.trim());
  const modelNameConflict =
    !!normalizedModelName &&
    takenModelNames.some(
      (n) => n.replace(/\.pt$/i, "").toLowerCase() === normalizedModelName.toLowerCase()
    );
  const modelNameInvalid = !normalizedModelName || modelNameConflict;
  const visibleJobs = jobs.filter((j) => {
    if (j.status === "queued" || j.status === "running" || j.status === "failed") {
      return true;
    }
    if (!j.model_name) return false;
    return models.some((m) => m.name === j.model_name);
  });

  const onTrain = async () => {
    if (!selectedDataset || modelNameInvalid) return;
    setTraining(true);
    setError(null);
    setInfo(null);
    try {
      const job = await api.startTrain({
        dataset_name: selectedDataset,
        epochs,
        base_model: baseModel,
        model_name: normalizedModelName,
      });
      setInfo(`Job ${job.id} ${job.status}`);
      const nextJobs = [job, ...jobs.filter((j) => j.id !== job.id)];
      setJobs(nextJobs);
      const taken = [
        ...models.map((m) => m.name),
        ...nextJobs.map((j) => j.model_name).filter((n): n is string => !!n),
      ];
      const nextName = suggestModelName(
        selectedMeta?.name || selectedDataset,
        baseModel,
        taken
      );
      setNameTouched(false);
      setParams({ name: nextName });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed to start");
    } finally {
      setTraining(false);
    }
  };

  const onPredict = async (file: File | null) => {
    if (!file || !testModel) return;
    if (!file.type.startsWith("image/")) {
      setError("Upload an image file");
      return;
    }
    setPredicting(true);
    setError(null);
    setInfo(null);
    setPredictResult(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
    try {
      const result = await api.predict(file, testModel);
      setPredictResult(result);
      setInfo(`Predicted with ${result.model}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Predict failed");
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="relative mx-auto flex min-h-screen max-w-3xl flex-col px-4 py-6 sm:px-6 sm:py-8">
      <header className="mb-8">
        <motion.div
          className="mb-4 h-1.5 w-16 origin-left bg-primary"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        />
        <div className="flex items-end gap-4">
          <motion.img
            src="/logo.jpg"
            alt="BlackLocust"
            width={56}
            height={56}
            className="size-14 shrink-0 object-cover"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.08 }}
          />
          <div className="min-w-0">
            <motion.h1
              className="font-display text-4xl leading-none tracking-tight text-foreground sm:text-5xl"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: 0.12 }}
            >
              BlackLocust
            </motion.h1>
            <motion.p
              className="mt-2 max-w-md text-sm text-muted-foreground"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.28 }}
            >
              An AI trainer powered by Yolo26
            </motion.p>
          </div>
        </div>
      </header>

      <nav className="mb-6 grid grid-cols-3 gap-2" aria-label="Steps">
        {(
          [
            { id: "dataset" as const, label: "Dataset", n: "01" },
            { id: "train" as const, label: "Train", n: "02" },
            { id: "test" as const, label: "Test", n: "03" },
          ] as const
        ).map((s) => {
          const active = step === s.id;
          const locked =
            (s.id === "train" && !selectedDataset) || (s.id === "test" && models.length === 0);
          return (
            <button
              key={s.id}
              type="button"
              disabled={locked}
              onClick={() => {
                if (s.id === "train" && selectedDataset) {
                  setParams({ step: "train", dataset: selectedDataset });
                } else if (s.id === "test" && models.length > 0) {
                  setParams({
                    step: "test",
                    testModel: testModel || models[0]?.name || null,
                  });
                } else if (s.id === "dataset") {
                  setParams({ step: "dataset" });
                }
              }}
              className={cn(
                "relative overflow-hidden border px-3 py-2.5 text-left transition-colors",
                active
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-border bg-card text-muted-foreground hover:border-primary/60 hover:text-foreground",
                locked && "cursor-not-allowed opacity-35 hover:border-border hover:text-muted-foreground"
              )}
            >
              {active && (
                <motion.span
                  layoutId="step-glow"
                  className="absolute inset-x-0 bottom-0 h-1 bg-foreground/20"
                  transition={{ type: "spring", stiffness: 380, damping: 32 }}
                />
              )}
              <span className="font-display text-xs tracking-wide opacity-70">{s.n}</span>
              <div className="mt-1 text-sm font-semibold">{s.label}</div>
            </button>
          );
        })}
      </nav>

      {(error || info) && (
        <div className="mb-5 flex items-start gap-3 border-l-4 border-primary bg-primary/15 px-3 py-2.5 text-sm">
          <div className="min-w-0 flex-1">
            {error && <p className="text-destructive">{error}</p>}
            {info && <p className="text-foreground">{info}</p>}
          </div>
          <button
            type="button"
            aria-label="Close"
            className="shrink-0 p-0.5 text-muted-foreground transition-colors hover:text-foreground"
            onClick={() => {
              setError(null);
              setInfo(null);
            }}
          >
            <X className="size-4" />
          </button>
        </div>
      )}

      <AnimatePresence mode="wait">
        {step === "dataset" ? (
          <motion.div key="dataset" {...stepMotion} className="space-y-6">
            <section className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="font-display text-2xl">Datasets</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Pick a dataset, or upload a new zip.
                  </p>
                </div>
                <Button type="button" size="lg" className="h-10 px-4" onClick={openUpload}>
                  Upload
                </Button>
              </div>
              {datasets.length === 0 ? (
                <p className="text-sm text-muted-foreground">No uploads yet.</p>
              ) : (
                <ul className="overflow-hidden border border-border bg-card">
                  {datasets.map((d, i) => {
                    const key = datasetKey(d);
                    const selected = selectedDataset === key;
                    return (
                      <motion.li
                        key={d.id ?? key ?? `dataset-${i}`}
                        initial={{ opacity: 0, x: -12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.05 }}
                        className={cn(
                          "grid grid-cols-[1fr_auto] items-stretch border-b border-border last:border-b-0",
                          selected ? "bg-primary/25" : "hover:bg-accent/40"
                        )}
                      >
                        <button
                          type="button"
                          onClick={() => setParams({ dataset: key })}
                          className="grid grid-cols-[auto_1fr] items-center gap-3 px-3.5 py-3 text-left"
                        >
                          <span
                            className={cn(
                              "size-2.5 shrink-0 rounded-full",
                              selected ? "bg-primary" : "bg-muted-foreground/40"
                            )}
                          />
                          <span className="min-w-0">
                            <span className="block truncate font-medium text-foreground">
                              {d.name}
                              {d.ready === false ? " · not ready" : ""}
                            </span>
                            <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                              {key}
                              {d.created_at
                                ? ` · ${new Date(d.created_at).toLocaleString()}`
                                : ""}
                            </span>
                          </span>
                        </button>
                        <button
                          type="button"
                          className="px-3.5 text-xs font-medium text-destructive hover:bg-destructive/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteDataset(d);
                          }}
                        >
                          Delete
                        </button>
                      </motion.li>
                    );
                  })}
                </ul>
              )}
            </section>

            <section className="space-y-3">
              <div>
                <h2 className="font-display text-2xl">Model</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Start from a pretrained YOLO weight, or continue from a trained model.
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  <span className="font-medium text-foreground">classify</span> — one label for
                  the whole image (e.g. cat vs dog). Use{" "}
                  <span className="font-medium text-foreground">*-cls</span>.
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  <span className="font-medium text-foreground">detect</span> — find objects and
                  draw boxes. Use models without{" "}
                  <span className="font-medium text-foreground">-cls</span>.
                </p>
              </div>
              <ul className="overflow-hidden border border-border bg-card">
                {(
                  [
                    { value: "yolo26n-cls.pt", label: "yolo26n-cls", hint: "pretrained · classify" },
                    { value: "yolo26s-cls.pt", label: "yolo26s-cls", hint: "pretrained · classify" },
                    { value: "yolo26m-cls.pt", label: "yolo26m-cls", hint: "pretrained · classify" },
                    { value: "yolo26n.pt", label: "yolo26n", hint: "pretrained · detect" },
                    { value: "yolo26s.pt", label: "yolo26s", hint: "pretrained · detect" },
                    { value: "yolo26m.pt", label: "yolo26m", hint: "pretrained · detect" },
                  ] as const
                ).map((m, i) => {
                  const selected = baseModel === m.value;
                  return (
                    <motion.li
                      key={m.value}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.03 }}
                    >
                      <button
                        type="button"
                        onClick={() =>
                          setParams({ model: selected ? null : m.value })
                        }
                        className={cn(
                          "grid w-full grid-cols-[auto_1fr] items-center gap-3 border-b border-border px-3.5 py-3 text-left transition-colors",
                          selected ? "bg-primary/25" : "hover:bg-accent/40"
                        )}
                      >
                        <span
                          className={cn(
                            "size-2.5 shrink-0 rounded-full",
                            selected ? "bg-primary" : "bg-muted-foreground/40"
                          )}
                        />
                        <span className="min-w-0">
                          <span className="block truncate font-medium text-foreground">
                            {m.label}
                          </span>
                          <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                            {m.hint}
                          </span>
                        </span>
                      </button>
                    </motion.li>
                  );
                })}
                {models.map((m, i) => {
                  const selected = baseModel === m.name;
                  return (
                    <motion.li
                      key={m.name}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: (6 + i) * 0.03 }}
                      className={cn(
                        "grid grid-cols-[1fr_auto] items-stretch border-b border-border last:border-b-0",
                        selected ? "bg-primary/25" : "hover:bg-accent/40"
                      )}
                    >
                      <button
                        type="button"
                        onClick={() =>
                          setParams({ model: selected ? null : m.name })
                        }
                        className="grid grid-cols-[auto_1fr] items-center gap-3 px-3.5 py-3 text-left"
                      >
                        <span
                          className={cn(
                            "size-2.5 shrink-0 rounded-full",
                            selected ? "bg-primary" : "bg-muted-foreground/40"
                          )}
                        />
                        <span className="min-w-0">
                          <span className="block truncate font-medium text-foreground">
                            {m.name}
                          </span>
                          <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                            trained · {(m.size_bytes / (1024 * 1024)).toFixed(2)} MB ·{" "}
                            {new Date(m.modified_at).toLocaleString()}
                          </span>
                        </span>
                      </button>
                      <button
                        type="button"
                        className="px-3.5 text-xs font-medium text-destructive hover:bg-destructive/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteModel(m.name);
                        }}
                      >
                        Delete
                      </button>
                    </motion.li>
                  );
                })}
              </ul>
            </section>

            <div className="mb-[10px] flex justify-end">
              <Button
                type="button"
                size="lg"
                className="group h-11 min-w-40 gap-2 px-5 font-display text-base"
                disabled={!selectedDataset || selectedMeta?.ready === false || !baseModel}
                onClick={() => {
                  setNameTouched(false);
                  const suggested = suggestModelName(
                    selectedMeta?.name || selectedDataset,
                    baseModel,
                    [
                      ...models.map((m) => m.name),
                      ...jobs.map((j) => j.model_name).filter((n): n is string => !!n),
                    ]
                  );
                  setParams(
                    {
                      step: "train",
                      dataset: selectedDataset,
                      model: baseModel,
                      name: suggested,
                    },
                    false
                  );
                }}
              >
                Continue
                <ArrowRight className="size-4 transition-transform duration-200 group-hover:translate-x-1" />
              </Button>
            </div>
          </motion.div>
        ) : step === "train" ? (
          <motion.div key="train" {...stepMotion} className="space-y-6">
            <section className="space-y-4">
              <h2 className="font-display text-2xl">Train</h2>
              <p className="text-sm text-muted-foreground">
                Dataset{" "}
                <span className="font-medium text-foreground">
                  {selectedMeta?.name ?? selectedDataset}
                </span>
                {" · "}
                Model{" "}
                <span className="font-medium text-foreground">{baseModel || "—"}</span>
              </p>
              <div className="grid w-full gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label htmlFor="model-name">Model name</Label>
                  <Input
                    id="model-name"
                    placeholder="cats_dogs_yolo26n-cls"
                    value={modelName}
                    aria-invalid={modelNameConflict || undefined}
                    onChange={(e) => {
                      setNameTouched(true);
                      setParams({ name: e.target.value || null });
                    }}
                  />
                  <p className="text-xs text-muted-foreground">
                    Saves as{" "}
                    <span className="font-medium text-foreground">
                      {normalizedModelName ? `${normalizedModelName}.pt` : "—.pt"}
                    </span>
                  </p>
                  {modelNameConflict && (
                    <p className="text-xs text-destructive">
                      A model with this name already exists. Choose another.
                    </p>
                  )}
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="epochs">Epochs</Label>
                  <Input
                    id="epochs"
                    type="number"
                    min={1}
                    max={500}
                    value={epochs}
                    onChange={(e) =>
                      setParams({ epochs: String(Number(e.target.value) || 1) })
                    }
                  />
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                <Button
                  type="button"
                  variant="outline"
                  size="lg"
                  className="h-11 px-4"
                  onClick={() => setParams({ step: "dataset" })}
                >
                  Back
                </Button>
                <Button
                  type="button"
                  size="lg"
                  className="h-11 min-w-40 gap-2 px-5 font-display text-base"
                  disabled={
                    training || !selectedDataset || running || modelNameInvalid
                  }
                  onClick={onTrain}
                >
                  {training || running ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Play className="size-4 fill-current" />
                  )}
                  {training ? "Starting…" : running ? "Training…" : "Start training"}
                </Button>
              </div>
            </section>

            <section className="mb-[10px] space-y-3">
              <h2 className="font-display text-2xl">
                Jobs{" "}
                <span className="text-muted-foreground">{visibleJobs.length}</span>
              </h2>
              {visibleJobs.length === 0 ? (
                <p className="text-sm text-muted-foreground">No training jobs yet.</p>
              ) : (
                <ul>
                  {visibleJobs.map((j, i) => (
                    <motion.li
                      key={j.id ?? `job-${i}`}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                      className="border-b border-border py-3 last:border-b-0"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="font-medium">
                          {j.id}
                        </span>
                        <span className="flex items-center gap-2">
                          {(j.status === "running" || j.status === "queued") && (
                            <span className="tabular-nums text-sm font-medium text-foreground">
                              {j.current_epoch ?? 0}/{j.epochs}
                            </span>
                          )}
                          {j.status === "completed" &&
                          j.model_name &&
                          models.some((m) => m.name === j.model_name) ? (
                            <Button
                              type="button"
                              size="sm"
                              className="h-8 px-3 font-display"
                              onClick={() =>
                                setParams({
                                  step: "test",
                                  testModel: j.model_name,
                                })
                              }
                            >
                              Test
                            </Button>
                          ) : j.status !== "completed" ? (
                            <span
                              className={cn(
                                "font-display text-xs tracking-wider uppercase",
                                j.status === "running" || j.status === "queued"
                                  ? "text-foreground"
                                  : j.status === "failed"
                                    ? "text-destructive"
                                    : "text-muted-foreground"
                              )}
                            >
                              {j.status === "running" || j.status === "queued" ? (
                                <motion.span
                                  animate={{ opacity: [1, 0.35, 1] }}
                                  transition={{ duration: 1.4, repeat: Infinity }}
                                >
                                  {j.status}
                                </motion.span>
                              ) : (
                                j.status
                              )}
                            </span>
                          ) : null}
                        </span>
                      </div>
                      <div className="mt-1 text-sm text-muted-foreground">
                        {j.dataset_name} · {j.base_model} · {j.epochs} epochs
                        {j.model_name ? ` · ${j.model_name}` : ""}
                      </div>
                      {(j.message || j.error) && (
                        <div
                          className={cn(
                            "mt-1 text-sm",
                            j.error
                              ? "text-destructive"
                              : j.status === "completed"
                                ? "text-emerald-600"
                                : "text-muted-foreground"
                          )}
                        >
                          {j.error || j.message}
                        </div>
                      )}
                    </motion.li>
                  ))}
                </ul>
              )}
            </section>
          </motion.div>
        ) : (
          <motion.div key="test" {...stepMotion} className="space-y-6">
            <section className="space-y-3">
              <h2 className="font-display text-2xl">Test</h2>
              <p className="text-sm text-muted-foreground">
                Pick a trained model, drop an image, see the prediction.
              </p>
              {models.length === 0 ? (
                <p className="text-sm text-muted-foreground">Train a model first.</p>
              ) : (
                <ul className="overflow-hidden border border-border bg-card">
                  {models.map((m, i) => {
                    const selected = testModel === m.name;
                    return (
                      <motion.li
                        key={m.name}
                        initial={{ opacity: 0, x: -12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        className={cn(
                          "grid grid-cols-[1fr_auto] items-stretch border-b border-border last:border-b-0",
                          selected ? "bg-primary/25" : "hover:bg-accent/40"
                        )}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setPredictResult(null);
                            setParams({ testModel: m.name });
                          }}
                          className="grid grid-cols-[auto_1fr] items-center gap-3 px-3.5 py-3 text-left"
                        >
                          <span
                            className={cn(
                              "size-2.5 shrink-0 rounded-full",
                              selected ? "bg-primary" : "bg-muted-foreground/40"
                            )}
                          />
                          <span className="min-w-0">
                            <span className="block truncate font-medium">{m.name}</span>
                            <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                              {(m.size_bytes / (1024 * 1024)).toFixed(2)} MB ·{" "}
                              {new Date(m.modified_at).toLocaleString()}
                            </span>
                          </span>
                        </button>
                        <button
                          type="button"
                          className="px-3.5 text-xs font-medium text-destructive hover:bg-destructive/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteModel(m.name);
                          }}
                        >
                          Delete
                        </button>
                      </motion.li>
                    );
                  })}
                </ul>
              )}
            </section>

            <section className="space-y-3">
              <h2 className="font-display text-2xl">Image</h2>
              <input
                id="yolo-predict-upload"
                type="file"
                accept="image/*"
                className="hidden"
                disabled={predicting || !testModel}
                onChange={(e) => {
                  onPredict(e.target.files?.[0] ?? null);
                  e.target.value = "";
                }}
              />
              <div
                role="button"
                tabIndex={0}
                aria-disabled={predicting || !testModel}
                onDragEnter={(e) => {
                  e.preventDefault();
                  if (!predicting && testModel) setPredictDrag(true);
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!predicting && testModel) setPredictDrag(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  if (e.currentTarget.contains(e.relatedTarget as Node)) return;
                  setPredictDrag(false);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  setPredictDrag(false);
                  if (predicting || !testModel) return;
                  onPredict(e.dataTransfer.files?.[0] ?? null);
                }}
                onClick={() => {
                  if (!predicting && testModel)
                    document.getElementById("yolo-predict-upload")?.click();
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    if (!predicting && testModel)
                      document.getElementById("yolo-predict-upload")?.click();
                  }
                }}
                className={cn(
                  "flex min-h-40 cursor-pointer flex-col items-center justify-center gap-2 border-2 border-dashed px-4 py-8 text-center transition-colors",
                  predictDrag
                    ? "border-primary bg-primary/30"
                    : "border-primary/50 bg-primary/5 hover:border-primary hover:bg-primary/15",
                  (predicting || !testModel) && "pointer-events-none opacity-50"
                )}
              >
                <p className="font-display text-lg">
                  {predicting
                    ? "Predicting…"
                    : predictDrag
                      ? "Drop it"
                      : "Drop image here"}
                </p>
                {!predicting && (
                  <p className="text-xs text-muted-foreground">or click to browse</p>
                )}
              </div>
            </section>

            {(previewUrl || predictResult) && (
              <section className="space-y-4">
                <h2 className="font-display text-2xl">Result</h2>
                <div className="grid gap-4 sm:grid-cols-2">
                  {(predictResult?.image_base64 || previewUrl) && (
                    <img
                      src={
                        predictResult?.image_base64
                          ? `data:image/jpeg;base64,${predictResult.image_base64}`
                          : previewUrl!
                      }
                      alt="Prediction"
                      className="w-full border border-border object-contain"
                    />
                  )}
                  <div className="space-y-2">
                    {predictResult?.predictions.length ? (
                      <ul className="space-y-2">
                        {predictResult.predictions.map((p, i) => (
                          <li
                            key={`${p.label}-${i}`}
                            className="flex items-baseline justify-between gap-3 border-b border-border py-2"
                          >
                            <span className="font-medium">{p.label}</span>
                            <span className="font-display text-sm text-muted-foreground">
                              {(p.confidence * 100).toFixed(1)}%
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      !predicting && (
                        <p className="text-sm text-muted-foreground">No predictions yet.</p>
                      )
                    )}
                  </div>
                </div>
              </section>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {uploadOpen && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <button
              type="button"
              aria-label="Close"
              className="absolute inset-0 bg-foreground/50"
              disabled={uploading}
              onClick={() => {
                if (!uploading) closeUpload();
              }}
            />
            <motion.div
              role="dialog"
              aria-modal="true"
              aria-labelledby="upload-dataset-title"
              className="relative z-10 w-full max-w-2xl border border-primary/40 bg-card p-4 shadow-lg sm:p-6"
              initial={{ opacity: 0, y: 16, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.98 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="mb-4 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h2 id="upload-dataset-title" className="font-display text-2xl">
                    Upload dataset
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Zip with train/ and val/ class folders.
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={uploading}
                  onClick={closeUpload}
                >
                  Close
                </Button>
              </div>

              <input
                id="yolo-dataset-upload"
                type="file"
                accept=".zip,application/zip"
                className="hidden"
                disabled={uploading}
                onChange={(e) => {
                  onUpload(e.target.files?.[0] ?? null);
                  e.target.value = "";
                }}
              />

              <div className="grid gap-4 sm:grid-cols-2 sm:items-stretch">
                <div className="flex flex-col justify-center space-y-2">
                  <Label htmlFor="dataset-name">Optional name</Label>
                  <Input
                    id="dataset-name"
                    placeholder="my_dataset"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                  />
                </div>
                <div
                  role="button"
                  tabIndex={0}
                  aria-disabled={uploading}
                  onDragEnter={(e) => {
                    e.preventDefault();
                    if (!uploading) setDragOver(true);
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    if (!uploading) setDragOver(true);
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
                    setDragOver(false);
                  }}
                  onDrop={onDrop}
                  onClick={() => {
                    if (!uploading) document.getElementById("yolo-dataset-upload")?.click();
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      if (!uploading) document.getElementById("yolo-dataset-upload")?.click();
                    }
                  }}
                  className={cn(
                    "flex min-h-36 cursor-pointer flex-col items-center justify-center gap-2 border-2 border-dashed px-4 py-8 text-center transition-colors sm:min-h-full",
                    dragOver
                      ? "border-primary bg-primary/30"
                      : "border-primary/50 bg-primary/5 hover:border-primary hover:bg-primary/15",
                    uploading && "pointer-events-none opacity-50"
                  )}
                >
                  <p className="font-display text-lg text-foreground">
                    {uploading ? "Uploading…" : dragOver ? "Drop it" : "Drop .zip here"}
                  </p>
                  {!uploading && (
                    <p className="text-xs text-muted-foreground">or click to browse</p>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="mt-0 flex flex-col items-center gap-2 border-t border-border pt-4 text-xs text-muted-foreground">
        <span>Powered by</span>
        <a
          href="https://www.ultralytics.com"
          target="_blank"
          rel="noopener noreferrer"
          className="opacity-80 transition-opacity hover:opacity-100"
        >
          <img
            src="/ultralytics.svg"
            alt="Ultralytics"
            width={140}
            height={25}
            className="h-5 w-auto"
          />
        </a>
      </footer>
    </div>
  );
}
