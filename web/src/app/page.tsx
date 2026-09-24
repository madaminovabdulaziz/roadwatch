import Link from "next/link";
import HomeView from "@/views/HomeView";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-10 sm:py-16">
      <p className="text-sm uppercase tracking-widest text-sky-400">WIUT Hackathon 2026 · computer vision</p>
      <h1 className="mt-3 max-w-3xl text-3xl font-semibold tracking-tight sm:text-5xl">
        Finds traffic violations in CCTV video and warns before a crash.
      </h1>
      <p className="mt-4 max-w-2xl text-zinc-400">
        One fixed road camera in, a list of timed events (14 classes) and a live accident-risk curve out. Open
        weights, runs offline on one GPU.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <Link href="/demo/" className="rounded-lg bg-sky-500 px-4 py-2 font-medium text-white hover:bg-sky-400">
          Try the demo
        </Link>
        <Link href="/approach/" className="rounded-lg border border-white/15 px-4 py-2 hover:bg-white/10">
          Read the approach
        </Link>
      </div>
      <HomeView />
    </main>
  );
}
