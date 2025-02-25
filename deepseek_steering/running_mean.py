import torch
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self):
        """
        Calculates the running mean, std, and sum of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = None
        self.var = None
        self.count = 0
        self.sum = None  # Add sum tracking

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd()
        if self.mean is not None:
            new_object.mean = self.mean.clone()
            new_object.var = self.var.clone()
            new_object.count = float(self.count)
            new_object.sum = self.sum.clone()  # Copy sum
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        """
        self.update_from_moments(other.mean, other.var, other.count)
        if self.sum is None:
            self.sum = other.sum.clone()
        else:
            self.sum += other.sum

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = arr.double().mean(dim=0)
        batch_var = arr.double().var(dim=0)
        batch_count = arr.shape[0]
        batch_sum = arr.double().sum(dim=0)  # Calculate batch sum
        
        if batch_count == 0:
            return
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            self.sum = batch_sum  # Initialize sum
        else:
            self.update_from_moments(batch_mean, batch_var, batch_count)
            self.sum += batch_sum  # Update sum

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def compute(
        self, return_dict=False
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor] | dict[str, float]:
        """
        Compute the running mean, variance, count and sum

        Returns:
            mean, var, count, sum if return_dict=False
            dict with keys 'mean', 'var', 'count', 'sum' if return_dict=True
        """
        if return_dict:
            return {
                "mean": self.mean.item(),
                "var": self.var.item(),
                "count": self.count,
                "sum": self.sum.item(),
            }
        return self.mean, self.var, self.count, self.sum