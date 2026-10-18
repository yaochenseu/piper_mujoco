"""Verify installed vision packages and CUDA kernels without model weights."""
import argparse
from contextlib import redirect_stdout
from importlib import import_module, metadata
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true", help="Skip execution of CUDA kernels")
    args = parser.parse_args()
    results = {"python": sys.executable, "environment": sys.prefix, "checks": {}}

    def check(name, function):
        try:
            with redirect_stdout(sys.stderr):
                details = function()
            results["checks"][name] = {"ok": True, "details": details}
        except Exception:
            results["checks"][name] = {"ok": False, "error": traceback.format_exc()}

    def packages():
        versions = {}
        for line in (ROOT / "requirements-piperx-vision.txt").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("==")[0]
            versions[name] = metadata.version(name)
        for name in ("cv2", "open3d", "PIL.Image", "ultralytics", "transformers", "yaml", "dotenv"):
            import_module(name)
        return versions

    def torch_cuda():
        import torch
        import torchvision
        info = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                "cuda": torch.version.cuda, "gpu_available": torch.cuda.is_available()}
        if not args.imports_only:
            assert torch.cuda.is_available(), "CUDA device is unavailable"
            info["gpu"] = torch.cuda.get_device_name(0)
            x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            torch.testing.assert_close(x @ torch.eye(4, device="cuda"), x)
            torch.cuda.synchronize()
        return info

    def pointnet():
        import torch
        extension = import_module("pointnet2._ext")
        sys.path.insert(0, str(ROOT / "graspnet-baseline/pointnet2"))
        from pointnet2_utils import furthest_point_sample, gather_operation
        if not args.imports_only:
            torch.manual_seed(7)
            points = torch.rand(1, 128, 3, device="cuda")
            indices = furthest_point_sample(points, 16)
            assert indices.shape == (1, 16)
            assert indices.min() >= 0 and indices.max() < 128
            assert indices.unique().numel() == 16
            values = points.transpose(1, 2).contiguous()
            gathered = gather_operation(values, indices)
            expected = torch.gather(values, 2, indices.long().unsqueeze(1).expand(-1, 3, -1))
            torch.testing.assert_close(gathered, expected)
            torch.cuda.synchronize()
        return extension.__file__

    def knn():
        import torch
        extension = import_module("knn_pytorch.knn_pytorch")
        if not args.imports_only:
            # This repository's KNN extension returns one-based indices.
            reference = torch.tensor([[[0., 2., 5.], [0., 0., 0.], [0., 0., 0.]]], device="cuda")
            query = torch.tensor([[[0.1, 4.8], [0., 0.], [0., 0.]]], device="cuda")
            indices = torch.empty((1, 1, 2), dtype=torch.int64, device="cuda")
            extension.knn(reference, query, indices)
            torch.testing.assert_close(indices.cpu(), torch.tensor([[[1, 3]]]))
            torch.cuda.synchronize()
        return extension.__file__

    def groundingdino():
        import torch
        from groundingdino.util.inference import Model  # noqa: F401
        extension = import_module("groundingdino._C")
        if not args.imports_only:
            # A one-cell deformable-attention sample must reproduce its value.
            from groundingdino.models.GroundingDINO.ms_deform_attn import MultiScaleDeformableAttnFunction
            value = torch.tensor([[[[2., 3., 5., 7.]]]], device="cuda")
            shapes = torch.tensor([[1, 1]], dtype=torch.long, device="cuda")
            starts = torch.tensor([0], dtype=torch.long, device="cuda")
            locations = torch.full((1, 1, 1, 1, 1, 2), 0.5, device="cuda")
            weights = torch.ones((1, 1, 1, 1, 1), device="cuda")
            actual = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, locations, weights, 1)
            torch.testing.assert_close(actual, value.reshape(1, 1, 4))
            torch.cuda.synchronize()
        return extension.__file__

    def grasp_api():
        import numpy as np
        from graspnetAPI import GraspGroup
        import graspnetAPI
        array = np.zeros((2, 17))
        array[:, 0] = [0.9, 0.8]
        array[:, 1:4] = [0.04, 0.02, 0.03]
        array[:, 4:13] = np.eye(3).reshape(1, 9)
        array[:, 16] = -1
        group = GraspGroup(array).nms()
        assert len(group) == 1, "Grasp NMS did not remove duplicate grasp"
        return {"module": graspnetAPI.__file__, "nms": "OK"}

    def model_import():
        import torch
        from utils.path_setup import setup_graspnet_path
        setup_graspnet_path()
        from graspnet import GraspNet, pred_decode
        torch.manual_seed(7)
        net = GraspNet(is_training=False).eval()
        info = {"parameters": sum(p.numel() for p in net.parameters())}
        if not args.imports_only:
            net = net.cuda()
            with torch.no_grad():
                outputs = net({"point_clouds": torch.rand(1, 3000, 3, device="cuda") * 0.1})
                predictions = pred_decode(outputs)[0]
            assert predictions.ndim == 2 and predictions.shape[1] == 17
            assert torch.isfinite(predictions).all()
            info["untrained_forward_shape"] = list(predictions.shape)
            torch.cuda.synchronize()
        return info

    check("packages", packages)
    check("torch_cuda", torch_cuda)
    check("pointnet2", pointnet)
    check("knn", knn)
    check("groundingdino", groundingdino)
    check("graspnet_api", grasp_api)
    check("graspnet_model", model_import)
    print(json.dumps(results, indent=2))
    return 0 if all(item["ok"] for item in results["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
