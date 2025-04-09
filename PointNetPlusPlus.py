import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def farthest_point_sample(xyz, npoint):
    """Improved Farthest Point Sampling with batch processing"""
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
    """
    centroids: (B, npoint, 3) - coordinates of the downsampled center point
    xyz: (B, N, 3) - coordinates of the original point cloud
    radius: radius of the sphere
    max_samples: maximum number of sampling points per region
    return:
    grouped_idx: (B, npoint, max_samples) - neighborhood index of each center point
    """
    B, npoint, _ = centroids.shape
    N = xyz.shape[1]

    # Calculate the distance from all points to the center point
    dist = torch.cdist(centroids, xyz)  # (B, npoint, N)

    # Find points with distance < radius and take the first max_samples
    mask = dist < radius
    grouped_idx = torch.zeros((B, npoint, max_samples), dtype=torch.long).to(xyz.device)

    for i in range(B):
        for j in range(npoint):
            valid_idx = torch.where(mask[i, j])[0]  # Index of points that meet the condition
            if len(valid_idx) > max_samples:
                valid_idx = valid_idx[:max_samples]  # Randomly or sequentially take the first max_samples
            grouped_idx[i, j, :len(valid_idx)] = valid_idx

    return grouped_idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, max_samples, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.max_samples = max_samples

        # MLP for local feature extraction
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
        """
        xyz: (B, N, 3) - point cloud coordinates
        points: (B, N, C) - point features (if it is the first layer, C=0)
        """
        B, N, _ = xyz.shape

        # 1. Farthest point sampling (FPS) to obtain the center point
        centroids_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        centroids = torch.gather(xyz, 1, centroids_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, npoint, 3)

        # 2. Ball Query Find Neighborhood
        grouped_idx = ball_query(centroids, xyz, self.radius, self.max_samples)  # (B, npoint, max_samples)
        grouped_xyz = torch.gather(xyz.unsqueeze(1).expand(-1, self.npoint, -1, -1),
                                   2,
                                   grouped_idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # (B, npoint, max_samples, 3)

        # 3. Normalized coordinates (relative to the center point)
        grouped_xyz -= centroids.unsqueeze(2)

        # 4. Stitching coordinates and features (if any)
        if points is not None:
            grouped_points = torch.gather(points.unsqueeze(1).expand(-1, self.npoint, -1, -1),
                                          2,
                                          grouped_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
            grouped_input = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, max_samples, 3+C)
        else:
            grouped_input = grouped_xyz

        # 5. Local feature extraction (MLP + max pooling)
        grouped_input = grouped_input.permute(0, 3, 1, 2)  # (B, 3+C, npoint, max_samples)
        new_points = self.mlp(grouped_input)  # (B, mlp[-1], npoint, max_samples)
        new_points = torch.max(new_points, 3)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])

        return centroids, new_points


# class PointNet2Classification(nn.Module):
#     def __init__(self, num_classes=40):
#         super().__init__()
#
#         # SA1: input (B, 1024, 3) → output (B, 512, 64)
#         self.sa1 = SetAbstraction(
#             npoint=512,
#             radius=0.2,
#             max_samples=32,
#             in_channels=3,
#             mlp_channels=[64, 64, 128]
#         )
#
#         # SA2: input (B, 512, 128) → output (B, 128, 256)
#         self.sa2 = SetAbstraction(
#             npoint=128,
#             radius=0.4,
#             max_samples=64,
#             in_channels=128 + 3,  # Input is coordinates + features
#             mlp_channels=[128, 128, 256]
#         )
#
#         # SA3: input (B, 128, 256) → output (B, 1, 1024)
#         self.sa3 = SetAbstraction(
#             npoint=1,  # The last layer only takes 1 point (global feature)
#             radius=0.8,
#             max_samples=128,
#             in_channels=256 + 3,
#             mlp_channels=[256, 512, 1024]
#         )
#
#
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(256, num_classes)
#
#     def forward(self, x):
#         B, _, _ = x.shape
#
#         # Hierarchical feature extraction
#         xyz = x
#         points = None  # The first layer has no features
#
#         xyz, points = self.sa1(xyz, points)  # (B, 512, 128)
#         xyz, points = self.sa2(xyz, points)  # (B, 128, 256)
#         xyz, points = self.sa3(xyz, points)  # (B, 1, 1024)
#
#         # global features
#         global_feat = points.view(B, -1)  # (B, 1024)
#
#
#         x = F.relu(self.bn1(self.fc1(global_feat)))
#         x = self.drop1(x)
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.drop2(x)
#         x = self.fc3(x)
#
#         return x


# # delete SA3
# class PointNet2Classification(nn.Module):
#     def __init__(self, num_classes=40):
#         super().__init__()
#
#         self.sa1 = SetAbstraction(
#             npoint=512,
#             radius=0.2,
#             max_samples=32,
#             in_channels=3,
#             mlp_channels=[64, 64, 128]
#         )
#
#         self.sa2 = SetAbstraction(
#             npoint=128,
#             radius=0.4,
#             max_samples=64,
#             in_channels=128 + 3,
#             mlp_channels=[128, 128, 256]
#         )
#
#         # Modify the classification head input dimension (from 1024 to 256)
#         self.fc1 = nn.Linear(256, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(256, num_classes)
#
#     def forward(self, x):
#         B, _, _ = x.shape
#         xyz = x
#         points = None
#
#         xyz, points = self.sa1(xyz, points)  # (B, 512, 128)
#         xyz, points = self.sa2(xyz, points)  # (B, 128, 256)
#
#         # Use the output of SA2 as global features (global maximum pooling)
#         global_feat = torch.max(points, 1)[0]  # (B, 256)
#
#
#         x = F.relu(self.bn1(self.fc1(global_feat)))
#         x = self.drop1(x)
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.drop2(x)
#         x = self.fc3(x)
#         return x

# delete SA2
class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()

        # SA1: input 1024 → output 512
        self.sa1 = SetAbstraction(
            npoint=512,
            radius=0.2,
            max_samples=32,
            in_channels=3,  # Enter only coordinates
            mlp_channels=[64, 64, 128]  # Output 128-dimensional features
        )

        # SA3: Directly receive the output of SA1 (512 points, 128-dimensional features)
        self.sa3 = SetAbstraction(
            npoint=1,  # global features
            radius=0.8,
            max_samples=128,
            in_channels=128 + 3,  # Output features + coordinates of SA1
            mlp_channels=[256, 512, 1024]  # Output 1024 dimensions
        )


        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        B, _, _ = x.shape
        xyz = x  # input point cloud (B, 1024, 3)
        points = None  # 初始无特征

        # SA1: Downsample to 512 points
        xyz, points = self.sa1(xyz, points)  # (B, 512, 128)

        # SA3: Aggregate directly from 512 points to global features
        xyz, points = self.sa3(xyz, points)  # (B, 1, 1024)
        global_feat = points.view(B, -1)  # (B, 1024)


        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


