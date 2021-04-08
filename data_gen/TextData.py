import torch


class TextData(torch.utils.data.Dataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, description, data):
        """Initiate the dataset abstraction.
        """
        super(TextData, self).__init__()
        self.data = data
        self.length=len(data)
        self.description = description

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        return self.description

    def __len__(self):
        return self.length
