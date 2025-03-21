from .arcs.mms import PCReduced_arc


class PCReduced(PCReduced_arc):
    """
    Load a train PCReduced model for MMS dayside plasma region clessification.

    Examples:
        >>> from spacephyml.models.mms import PCReduced
        >>> model = PCReduced('s84')

    Args:
        model (string): The model to load, ['s42', 's84', 's168', 's336']
        path (string): Path to loacation of stored model.
    """

    _models = {'s42': {
                    'url': '',
                    'file': 'model_PCReduced_s42.pth'},
               's84': {
                    'url': '',
                    'file': 'model_PCReduced_s84.pth'},
               's168': {
                    'url': '',
                    'file': 'model_PCReduced_s168.pth'},
               's336': {
                    'url': '',
                    'file': 'model_PCReduced_s336.pth'},
               }

    def __init__(self, model='42', path = './model'):
        super().__init__()

        if model not in self._models.keys():
            raise ValueError(f'Incorrect model, {model} not in' +
                             f'{self._models.keys()}')

        filepath = f'{path}/' + self._models[model]['file']

        missing = missing_files([filepath], './')
        if missing:
            print('Missing dataset file, downloading')
            makedirs(path, exist_ok=True)
            download_file_with_status(self._models[model]['url'], filepath)

        self.classifier.load_state_dict(torch.load(filepath, weights_only=True))
