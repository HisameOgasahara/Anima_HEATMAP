"""Streaming reduction with full spatial and selected token resolution."""


class RunningMap:
    def __init__(self, record, aggregation):
        self.record = record
        self.aggregation = aggregation
        self.total = None
        self.pending = None
        self.call = record["call"]
        self.call_count = 0
        self.count = 0

    def validate(self, record):
        for key in ("shape", "grid", "valid_grid", "query_indices", "key_indices", "head_ids", "total_heads"):
            if record[key] != self.record[key]:
                raise ValueError("통합 수집 중 지도 구조가 바뀌었습니다. keep_records=True로 수집하세요.")
        if self.aggregation == "daam" and record["call"] < self.call:
            raise ValueError("DAAM 통합 수집은 호출 순서를 유지해야 합니다.")

    def _flush(self):
        if self.pending is not None:
            self.pending /= self.call_count
            if self.total is None:
                self.total = self.pending
            else:
                self.total += self.pending
            self.pending = None
            self.call_count = 0

    def add(self, values, record):
        self.validate(record)
        if self.aggregation == "mean":
            if self.total is None:
                self.total = values
            else:
                self.total += values
        else:
            if record["call"] != self.call:
                self._flush()
                self.call = record["call"]
            if self.pending is None:
                self.pending = values
            else:
                self.pending += values
            self.call_count += 1
        self.count += 1

    def finish(self):
        if self.aggregation == "mean":
            self.total /= self.count
        else:
            self._flush()
        return self.total
