from typing import Callable, Tuple, List

class PromptOrRenderingScheduler:
    def __init__(self, 
                groupids_list: List[List[int]],
                goto_global_list: List[bool],
                ):
        self.groupids_list = groupids_list
        self.goto_global_list = goto_global_list
        assert len(groupids_list) == len(goto_global_list)
        self.custom_scheduler = self.convert_groupids_list_to_custom_scheduler(groupids_list)

    def get_current_ids(self, step) -> Tuple[int, ...]:
        return self.custom_scheduler(step)
    
    def is_goto_global(self, step) -> bool:
        return self.goto_global_list[step % len(self.goto_global_list)]
    
    @staticmethod
    def convert_groupids_list_to_custom_scheduler(groupids_list: List[List[int]]) -> Callable[[int], Tuple[int, ...]]:
        def custom_scheduler(step: int) -> Tuple[int, ...]:
            return tuple(groupids_list[step % len(groupids_list)])
        return custom_scheduler
    
    @classmethod
    def convert_groupids_list_to_PromptOrRenderingScheduler(cls, 
            object_groupids_list: List[List[int]], 
            single_object_repeat_n: int = 6,
        ) -> 'PromptOrRenderingScheduler':
        import itertools

        combinations = list(itertools.combinations(range(len(object_groupids_list)), 2))

        groupids_list = []
        groupids_list.extend(object_groupids_list * single_object_repeat_n)
        goto_global_list = [False] * len(groupids_list)
        
        for combo in combinations:
            merged_groupids = list(set(object_groupids_list[combo[0]] + object_groupids_list[combo[1]]))
            groupids_list.append(merged_groupids)
            goto_global_list.append(True)
        
        global_groupids = list(set(itertools.chain(*object_groupids_list)))
        groupids_list.append(global_groupids)
        goto_global_list.append(True)
        
        return cls(groupids_list, goto_global_list)