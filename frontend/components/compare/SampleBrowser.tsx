"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import {
  Search,
  ChevronLeft,
  ChevronRight,
  Database,
  User,
  Music,
  Gauge,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

interface LibriTTSSample {
  id: string;
  text: string;
  speaker_id: string;
  gender: string;
  pitch: string;
  speaking_rate: string;
  speech_monotony: string;
  utterance_pitch_mean: number;
  utterance_pitch_std: number;
}

interface SampleBrowserProps {
  onSelectSample: (sample: LibriTTSSample) => void;
  selectedSampleId?: string;
}

export function SampleBrowser({
  onSelectSample,
  selectedSampleId,
}: SampleBrowserProps) {
  const [samples, setSamples] = useState<LibriTTSSample[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState({
    gender: "",
    pitch: "",
    rate: "",
  });

  const limit = 20;

  const fetchSamples = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
      });

      if (search) params.set("search", search);
      if (filters.gender) params.set("gender", filters.gender);
      if (filters.pitch) params.set("pitch_category", filters.pitch);
      if (filters.rate) params.set("speaking_rate", filters.rate);

      const res = await fetch(`${API_URL}/libritts/samples?${params}`);
      const data = await res.json();

      setSamples(data.samples);
      setTotal(data.total);
    } catch (err) {
      console.error("Failed to fetch samples:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSamples();
  }, [offset, filters]);

  const handleSearch = () => {
    setOffset(0);
    fetchSamples();
  };

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-blue-400" />
            LibriTTS-R Samples
          </CardTitle>
          <Badge variant="outline" className="text-zinc-400">
            {total.toLocaleString()} samples
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Search and Filters */}
        <div className="flex gap-2 flex-wrap">
          <div className="flex-1 min-w-[200px] relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
            <Input
              placeholder="Search text..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="pl-9 bg-zinc-800 border-zinc-700 text-white"
            />
          </div>

          <Select
            value={filters.gender}
            onValueChange={(v) => setFilters((f) => ({ ...f, gender: v }))}
          >
            <SelectTrigger className="w-[120px] bg-zinc-800 border-zinc-700 text-white">
              <SelectValue placeholder="Gender" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="male">Male</SelectItem>
              <SelectItem value="female">Female</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={filters.pitch}
            onValueChange={(v) => setFilters((f) => ({ ...f, pitch: v }))}
          >
            <SelectTrigger className="w-[150px] bg-zinc-800 border-zinc-700 text-white">
              <SelectValue placeholder="Pitch" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="very low pitch">Very Low</SelectItem>
              <SelectItem value="quite low pitch">Quite Low</SelectItem>
              <SelectItem value="moderate pitch">Moderate</SelectItem>
              <SelectItem value="slightly high pitch">Slightly High</SelectItem>
              <SelectItem value="quite high pitch">Quite High</SelectItem>
              <SelectItem value="very high pitch">Very High</SelectItem>
            </SelectContent>
          </Select>

          <Button
            size="sm"
            onClick={handleSearch}
            className="bg-blue-600 hover:bg-blue-700"
          >
            Search
          </Button>
        </div>

        {/* Sample List */}
        <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
          {loading ? (
            <div className="text-center py-8 text-zinc-500">Loading...</div>
          ) : samples.length === 0 ? (
            <div className="text-center py-8 text-zinc-500">No samples found</div>
          ) : (
            samples.map((sample) => (
              <div
                key={sample.id}
                onClick={() => onSelectSample(sample)}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedSampleId === sample.id
                    ? "border-orange-500 bg-orange-500/10"
                    : "border-zinc-700 bg-zinc-800/50 hover:border-zinc-600"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white truncate">{sample.text}</p>
                    <div className="flex items-center gap-3 mt-2 text-xs text-zinc-500">
                      <span className="flex items-center gap-1">
                        <User className="w-3 h-3" />
                        {sample.speaker_id}
                      </span>
                      <span className="flex items-center gap-1">
                        <Music className="w-3 h-3" />
                        {sample.utterance_pitch_mean.toFixed(0)}Hz
                      </span>
                      <span className="flex items-center gap-1">
                        <Gauge className="w-3 h-3" />
                        {sample.speaking_rate}
                      </span>
                    </div>
                  </div>
                  <div className="flex flex-col gap-1 items-end">
                    <Badge
                      variant="outline"
                      className={
                        sample.gender === "male"
                          ? "text-blue-400 border-blue-400/50"
                          : "text-pink-400 border-pink-400/50"
                      }
                    >
                      {sample.gender}
                    </Badge>
                    <Badge variant="outline" className="text-zinc-400">
                      {sample.pitch}
                    </Badge>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between pt-2 border-t border-zinc-800">
          <span className="text-sm text-zinc-500">
            Page {currentPage} of {totalPages}
          </span>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0}
              className="border-zinc-700"
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setOffset(offset + limit)}
              disabled={offset + limit >= total}
              className="border-zinc-700"
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
