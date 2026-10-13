use super::*;
impl SecaEngine {
    pub(super) fn fnv1a_64(input: &str) -> i64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut hash = FNV_OFFSET;
        for b in input.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash as i64
    }
    pub(super) fn stable_source_id(&mut self, url: &str) -> i64 {
        if let Some(existing) = self.source_id_by_url.get(url) {
            return *existing;
        }

        let mut attempt = 0usize;
        loop {
            let candidate = if attempt == 0 {
                Self::fnv1a_64(url)
            } else {
                Self::fnv1a_64(&format!("{url}#{attempt}"))
            };

            match self.baseline_source_legend.get(&candidate) {
                None => {
                    self.source_id_by_url.insert(url.to_string(), candidate);
                    self.baseline_source_legend
                        .insert(candidate, url.to_string());
                    return candidate;
                }
                Some(existing_url) if existing_url == url => {
                    self.source_id_by_url.insert(url.to_string(), candidate);
                    return candidate;
                }
                Some(_) => {
                    attempt += 1;
                }
            }
        }
    }
    pub(super) fn register_batch_sources(&mut self, batch: &SourceBatch) {
        for source in &batch.sources {
            self.register_source_for_batch(source.source_id.as_str(), batch.batch_index);
        }
    }

    pub(super) fn register_source_for_batch(
        &mut self,
        external_source_id: &str,
        batch_index: u32,
    ) -> i64 {
        let internal_source_id = self.stable_source_id(external_source_id);
        self.baseline_source_legend
            .insert(internal_source_id, external_source_id.to_string());
        self.source_ids_by_batch_index
            .entry(batch_index)
            .or_default()
            .insert(internal_source_id);
        internal_source_id
    }
}
