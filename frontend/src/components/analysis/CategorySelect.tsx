import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { POLITICAL_CATEGORIES, type PoliticalCategory } from "@/lib/types";

export function CategorySelect({
  value,
  onChange,
}: {
  value: PoliticalCategory;
  onChange: (value: PoliticalCategory) => void;
}) {
  return (
    <div className="space-y-2">
      <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Content category (optional)
      </Label>
      <Select value={value} onValueChange={(v) => onChange(v as PoliticalCategory)}>
        <SelectTrigger className="w-full sm:w-[280px]">
          <SelectValue placeholder="Select a category" />
        </SelectTrigger>
        <SelectContent>
          {POLITICAL_CATEGORIES.map((c) => (
            <SelectItem key={c} value={c}>
              {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        Categories describe the content type only. No party, ideology or preference labelling is
        performed.
      </p>
    </div>
  );
}
