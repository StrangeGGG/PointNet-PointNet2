import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def ball_query(centroids, xyz, radius, max_samples):
    B, npoint, _ = centroids.shape
    N = xyz.shape[1]

    dist = torch.cdist(centroids, xyz)  # (B, npoint, N)
    mask = dist < radius
    grouped_idx = torch.zeros((B, npoint, max_samples), dtype=torch.long).to(xyz.device)

    for i in range(B):
        for j in range(npoint):
            valid_idx = torch.where(mask[i, j])[0]
            if len(valid_idx) > max_samples:
                valid_idx = valid_idx[:max_samples]
            grouped_idx[i, j, :len(valid_idx)] = valid_idx

    return grouped_idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, max_samples, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.max_samples = max_samples

        self.mlp = nn.Sequential()
        for i in range(len(mlp_channels)):
            self.mlp.add_module(
                f"conv_{i}",
                nn.Conv2d(in_channels if i == 0 else mlp_channels[i - 1],
                          mlp_channels[i],
                          kernel_size=1)
            )
            self.mlp.add_module(f"bn_{i}", nn.BatchNorm2d(mlp_channels[i]))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())

    def forward(self, xyz, points):
        B, N, _ = xyz.shape

        centroids_idx = farthest_point_sample(xyz, self.npoint)
        centroids = torch.gather(xyz, 1, centroids_idx.unsqueeze(-1).expand(-1, -1, 3))

        grouped_idx = ball_query(centroids, xyz, self.radius, self.max_samples)
        grouped_xyz = torch.gather(xyz.unsqueeze(1).expand(-1, self.npoint, -1, -1),
                                   2,
                                   grouped_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        grouped_xyz -= centroids.unsqueeze(2)

        if points is not None:
            grouped_points = torch.gather(points.unsqueeze(1).expand(-1, self.npoint, -1, -1),
                                          2,
                                          grouped_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
            grouped_input = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_input = grouped_xyz

        grouped_input = grouped_input.permute(0, 3, 1, 2)
        new_points = self.mlp(grouped_input)
        new_points = torch.max(new_points, 3)[0]
        new_points = new_points.permute(0, 2, 1)

        return centroids, new_points


class PointNet2OnlySA1(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()

        self.sa1 = SetAbstraction(
            npoint=512,
            radius=0.2,
            max_samples=32,
            in_channels=3,
            mlp_channels=[64, 64, 128]
        )

        self.fc1 = nn.Linear(128, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        B, _, _ = x.shape

        xyz = x
        points = None

        xyz, points = self.sa1(xyz, points)  # (B, 512, 128)
        global_feat = torch.max(points, dim=1)[0]  # (B, 128)

        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x
