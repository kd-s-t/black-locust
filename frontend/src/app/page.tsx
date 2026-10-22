import { Suspense } from "react";
import YoloTrain from "@/components/YoloTrain";

export default function Home() {
  return (
    <Suspense fallback={<div className="px-4 py-10 text-sm text-muted-foreground">Loading...</div>}>
      <YoloTrain />
    </Suspense>
  );
}
