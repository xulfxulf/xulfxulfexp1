from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    if (num_rel == 0).any():
        raise RuntimeError("at least one query has no relevant gallery image")
    tmp_cmc = matches.cumsum(1)
    inp = [
        tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100
    tmp_cmc = torch.stack(
        [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])], 1
    ) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100
    return all_cmc, mAP, mINP, indices


class Evaluator(object):
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger("IRRA.eval")

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        qids, gids, qfeats, gfeats = [], [], [], []
        for text_batch in self.txt_loader:
            if isinstance(text_batch, dict):
                pid = text_batch["pids"]
                caption = text_batch["caption_ids"].to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(
                        caption,
                        phrase_token_mask=text_batch["phrase_token_mask"].to(device),
                        phrase_valid_mask=text_batch["phrase_valid_mask"].to(device),
                    )
            else:
                pid, caption = text_batch
                caption = caption.to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(caption)
            qids.append(pid.view(-1))
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1))
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return qfeats, gfeats, qids, gids

    def _compute_hire_representations(self, model):
        actual = self._unwrap(model)
        model.eval()
        device = next(actual.parameters()).device
        qids, gids = [], []
        text_parts = {"mean": [], "variance": [], "state": []}
        image_parts = {"mean": [], "variance": [], "state": []}

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                encoded = actual.encode_text_retrieval(caption)
            qids.append(pid.view(-1).cpu())
            for key in text_parts:
                text_parts[key].append(encoded[key].float().cpu())

        for pid, image in self.img_loader:
            image = image.to(device)
            with torch.no_grad():
                encoded = actual.encode_image_retrieval(image)
            gids.append(pid.view(-1).cpu())
            for key in image_parts:
                image_parts[key].append(encoded[key].float().cpu())

        text_repr = {key: torch.cat(values, dim=0) for key, values in text_parts.items()}
        image_repr = {key: torch.cat(values, dim=0) for key, values in image_parts.items()}
        return text_repr, image_repr, torch.cat(qids), torch.cat(gids)

    def _compute_hire_v2_state_representations(self, model):
        """Collect support-free identity and state representations for v16.3.0."""
        actual = self._unwrap(model)
        model.eval()
        device = next(actual.parameters()).device
        qids, gids = [], []
        text_parts = {
            "identity_final": [],
            "state_tokens": [],
            "state_mask": [],
            "state_weights": [],
        }
        image_parts = {
            "identity_final": [],
            "state_tokens": [],
            "state_mask": [],
        }

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                encoded = actual.encode_text_state_retrieval(caption)
            qids.append(pid.view(-1).cpu())
            text_parts["identity_final"].append(
                encoded["identity_final"].float().cpu()
            )
            text_parts["state_tokens"].append(
                encoded["state_tokens"].float().cpu()
            )
            text_parts["state_mask"].append(
                encoded["state_mask"].bool().cpu()
            )
            text_parts["state_weights"].append(
                encoded["state_weights"].float().cpu()
            )

        for pid, image in self.img_loader:
            image = image.to(device)
            with torch.no_grad():
                encoded = actual.encode_image_state_retrieval(image)
            gids.append(pid.view(-1).cpu())
            image_parts["identity_final"].append(
                encoded["identity_final"].float().cpu()
            )
            image_parts["state_tokens"].append(
                encoded["state_tokens"].float().cpu()
            )
            image_parts["state_mask"].append(
                encoded["state_mask"].bool().cpu()
            )

        text_repr = {
            key: torch.cat(values, dim=0)
            for key, values in text_parts.items()
        }
        image_repr = {
            key: torch.cat(values, dim=0)
            for key, values in image_parts.items()
        }
        return (
            text_repr,
            image_repr,
            torch.cat(qids),
            torch.cat(gids),
        )

    @staticmethod
    def _format_table(rows):
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        for row in rows:
            table.add_row(row)
        for key in ("R1", "R5", "R10", "mAP", "mINP"):
            table.custom_format[key] = lambda _field, value: "{:.3f}".format(value)
        return table

    def eval(self, model, i2t_metric=False):
        actual = self._unwrap(model)

        if getattr(actual, "is_hire_v2_state_model", False):
            (
                text_repr,
                image_repr,
                qids,
                gids,
            ) = self._compute_hire_v2_state_representations(model)
            matrices = actual.compute_state_reranked_similarity(
                text_repr=text_repr,
                image_repr=image_repr,
                query_chunk=int(
                    getattr(actual.args, "hire_eval_query_chunk", 128)
                ),
            )
            rows = []
            results = {}
            for name in ("identity_final", "state_final"):
                cmc, mean_ap, mean_inp, order = rank(
                    matrices[name],
                    qids,
                    gids,
                    max_rank=10,
                    get_mAP=True,
                )
                del order
                result = {
                    "R1": float(cmc[0]),
                    "R5": float(cmc[4]),
                    "R10": float(cmc[9]),
                    "mAP": float(mean_ap),
                    "mINP": float(mean_inp),
                }
                results[name] = result
                rows.append([
                    "v16.3-" + name,
                    result["R1"],
                    result["R5"],
                    result["R10"],
                    result["mAP"],
                    result["mINP"],
                ])
            self.logger.info("\n" + str(self._format_table(rows)))
            self.logger.info(
                "v16.3 state gate: %.6f, rerank top-K: %d",
                float(actual.state_gate().detach().cpu()),
                int(actual.state_topk),
            )
            return results["state_final"]["R1"]

        if getattr(actual, "is_hire_model", False):
            text_repr, image_repr, qids, gids = self._compute_hire_representations(model)
            similarity = actual.compute_similarity_matrix(
                text_repr,
                image_repr,
                query_chunk=int(getattr(actual.args, "hire_eval_query_chunk", 128)),
                gallery_chunk=int(getattr(actual.args, "hire_eval_gallery_chunk", 512)),
            )
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
                similarity, qids, gids, max_rank=10, get_mAP=True
            )
            rows = [[
                "HIRE-t2i",
                float(t2i_cmc[0]),
                float(t2i_cmc[4]),
                float(t2i_cmc[9]),
                float(t2i_mAP),
                float(t2i_mINP),
            ]]
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                    similarity.t(), gids, qids, max_rank=10, get_mAP=True
                )
                rows.append([
                    "HIRE-i2t",
                    float(i2t_cmc[0]),
                    float(i2t_cmc[4]),
                    float(i2t_cmc[9]),
                    float(i2t_mAP),
                    float(i2t_mINP),
                ])
            self.logger.info("\n" + str(self._format_table(rows)))
            return float(t2i_cmc[0])

        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)
        similarity = qfeats @ gfeats.t()
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
            similarity, qids, gids, max_rank=10, get_mAP=True
        )
        rows = [[
            "t2i",
            float(t2i_cmc[0]),
            float(t2i_cmc[4]),
            float(t2i_cmc[9]),
            float(t2i_mAP),
            float(t2i_mINP),
        ]]
        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                similarity.t(), gids, qids, max_rank=10, get_mAP=True
            )
            rows.append([
                "i2t",
                float(i2t_cmc[0]),
                float(i2t_cmc[4]),
                float(i2t_cmc[9]),
                float(i2t_mAP),
                float(i2t_mINP),
            ])
        self.logger.info("\n" + str(self._format_table(rows)))
        return float(t2i_cmc[0])
