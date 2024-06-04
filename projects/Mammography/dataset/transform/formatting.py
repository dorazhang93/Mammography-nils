from collections import defaultdict
from mmpretrain.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmpretrain.datasets.transforms import PackInputs
from mmpretrain.structures import MultiTaskDataSample

@TRANSFORMS.register_module()
class MammoPackMultiTaskInputs(BaseTransform):
    """Convert all image labels of multi-task dataset to a dict of tensor.

    Args:
        multi_task_fields (Sequence[str]):
        input_key (str):
        task_handlers (dict):
    """

    def __init__(self,
                 multi_task_fields,
                 input_key='img',
                 task_handlers=dict()):
        """
        Args:
            multi_task_fields ('gt_label',):
            input_key ():
            task_handlers dict(N=PackInputs(algorithm_keys=['clinic_vars']),
                                LVI=PackInputs(algorithm_keys=['clinic_vars']),
                                multifocality=PackInputs(),
                                NumPos=PackInputs(),
                                tumor_size=PackInputs()):
        """
        self.multi_task_fields = multi_task_fields
        self.input_key = input_key
        self.task_handlers = defaultdict()
        for task_name, task_handler in task_handlers.items():
            self.task_handlers[task_name] = TRANSFORMS.build(task_handler)

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        result = {'img_path': 'a.png', 'gt_label': {'task1': 1, 'task3': 3},
            'img': array([[[  0,   0,   0])
        """
        packed_results = dict()
        results = results.copy()

        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = PackInputs.format_input(input_)

        task_results = defaultdict(dict)
        for field in self.multi_task_fields:
            if field in results:
                value = results.pop(field)
                for k, v in value.items():
                    task_results[k].update({field: v})

        results.pop('img')
        data_sample = MultiTaskDataSample()

        for task_name, task_result in task_results.items():
            task_handler = self.task_handlers[task_name]
            task_pack_result = task_handler({**results, **task_result})
            data_sample.set_field(task_pack_result['data_samples'], task_name)


        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self):
        repr = self.__class__.__name__
        task_handlers = ', '.join(
            f"'{name}': {handler.__class__.__name__}"
            for name, handler in self.task_handlers.items())
        repr += f'(multi_task_fields={self.multi_task_fields}, '
        repr += f"input_key='{self.input_key}', "
        repr += f'task_handlers={{{task_handlers}}})'
        return repr
