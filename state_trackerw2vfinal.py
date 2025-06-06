from db_query import DBQuery
import numpy as np
from utils import convert_list_to_dict
from dialogue_config import all_intents, all_slots, usersim_default_key
import copy
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize


class StateTracker:
    """Tracks the state of the episode/conversation and prepares the state representation for the agent."""

    def __init__(self, database, constants):
        self.db_helper = DBQuery(database)
        self.match_key = usersim_default_key
        self.intents_dict = convert_list_to_dict(all_intents)
        self.num_intents = len(all_intents)
        self.slots_dict = convert_list_to_dict(all_slots)
        self.num_slots = len(all_slots)
        self.max_round_num = constants['run']['max_round_num']
        self.none_state = np.zeros(self.get_state_size())
        self.word2vec_model = KeyedVectors.load_word2vec_format(
            '/content/drive/MyDrive/ArewardShap/ArewardShap/ArewardShap/GO-Bot-DRL/GoogleNews-vectors-negative300.bin.gz',
            binary=True
        )
        self.reset()

    def get_state_size(self):
        return 2 * self._get_single_state_size()

    def _get_single_state_size(self):
        return 2 * self.num_intents + 7 * self.num_slots + 3 + self.max_round_num

    def reset(self):
        self.current_informs = {}
        self.history = []
        self.round_num = 0
        self.goal = None

    def print_history(self):
        for action in self.history:
            print(action)

    def get_state(self, done=False):
        if done:
            return self.none_state

        user_action = self.history[-1]
        db_results_dict = self.db_helper.get_db_results_for_slots(self.current_informs)
        last_agent_action = self.history[-2] if len(self.history) > 1 else None

        user_act_rep = np.zeros((self.num_intents,))
        user_act_rep[self.intents_dict[user_action['intent']]] = 1.0

        user_inform_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['inform_slots'].keys():
            user_inform_slots_rep[self.slots_dict[key]] = 1.0

        user_request_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['request_slots'].keys():
            user_request_slots_rep[self.slots_dict[key]] = 1.0

        current_slots_rep = np.zeros((self.num_slots,))
        for key in self.current_informs:
            current_slots_rep[self.slots_dict[key]] = 1.0

        agent_act_rep = np.zeros((self.num_intents,))
        if last_agent_action:
            agent_act_rep[self.intents_dict[last_agent_action['intent']]] = 1.0

        agent_inform_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['inform_slots'].keys():
                agent_inform_slots_rep[self.slots_dict[key]] = 1.0

        agent_request_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['request_slots'].keys():
                agent_request_slots_rep[self.slots_dict[key]] = 1.0

        turn_rep = np.zeros((1,)) + self.round_num / 5.

        turn_onehot_rep = np.zeros((self.max_round_num,))
        if self.round_num > 0:
            turn_onehot_rep[self.round_num - 1] = 1.0

        kb_count_rep = np.zeros((self.num_slots + 1,)) + db_results_dict['matching_all_constraints'] / 100.
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_count_rep[self.slots_dict[key]] = db_results_dict[key] / 100.

        kb_binary_rep = np.zeros((self.num_slots + 1,))
        kb_binary_rep += float(db_results_dict['matching_all_constraints'] > 0.)
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_binary_rep[self.slots_dict[key]] = float(db_results_dict[key] > 0.)

        state_part = [
            user_act_rep, user_inform_slots_rep, user_request_slots_rep,
            agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep,
            current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep
        ]

        state = np.hstack(state_part)

        # مکانیزم توجه بر اساس هدف
        weighted_goal_embedding = self.get_attention_weighted_goal(state)

        # اتصال حالت با هدف وزن‌دار شده
        full_state = np.hstack([state, weighted_goal_embedding])
        return full_state

    def update_state_agent(self, agent_action):
        if agent_action['intent'] == 'inform':
            assert agent_action['inform_slots']
            inform_slots = self.db_helper.fill_inform_slot(agent_action['inform_slots'], self.current_informs)
            agent_action['inform_slots'] = inform_slots
            key, value = list(agent_action['inform_slots'].items())[0]
            assert key != 'match_found'
            assert value != 'PLACEHOLDER', 'KEY: {}'.format(key)
            self.current_informs[key] = value
        elif agent_action['intent'] == 'match_found':
            assert not agent_action['inform_slots'], 'Cannot inform and have intent of match found!'
            db_results = self.db_helper.get_db_results(self.current_informs)
            if db_results:
                key, value = list(db_results.items())[0]
                agent_action['inform_slots'] = copy.deepcopy(value)
                agent_action['inform_slots'][self.match_key] = str(key)
            else:
                agent_action['inform_slots'][self.match_key] = 'no match available'
            self.current_informs[self.match_key] = agent_action['inform_slots'][self.match_key]
        agent_action.update({'round': self.round_num, 'speaker': 'Agent'})
        self.history.append(agent_action)

    def update_state_user(self, user_action):
        for key, value in user_action['inform_slots'].items():
            self.current_informs[key] = value
        user_action.update({'round': self.round_num, 'speaker': 'User'})
        self.history.append(user_action)
        self.round_num += 1
        self.goal = user_action  # ذخیره هدف

    def get_goal_embedding(self):
        """
        Creates a goal embedding from the stored user goal.
        Returns:
            numpy.array: A numpy array of the same shape as _get_single_state_size().
        """
        if not self.goal:
            return np.zeros(self._get_single_state_size())

        user_act_rep = np.zeros((self.num_intents,))
        user_act_rep[self.intents_dict[self.goal['intent']]] = 1.0

        user_inform_slots_rep = np.zeros((self.num_slots,))
        for key in self.goal['inform_slots'].keys():
            user_inform_slots_rep[self.slots_dict[key]] = 1.0

        user_request_slots_rep = np.zeros((self.num_slots,))
        for key in self.goal['request_slots'].keys():
            user_request_slots_rep[self.slots_dict[key]] = 1.0

        empty_vec = np.zeros((self.num_intents,))
        empty_slots = np.zeros((self.num_slots,))
        turn_rep = np.zeros((1,))
        turn_onehot_rep = np.zeros((self.max_round_num,))
        kb_binary_rep = np.zeros((self.num_slots + 1,))
        kb_count_rep = np.zeros((self.num_slots + 1,))

        goal_embedding = np.hstack([
            user_act_rep, user_inform_slots_rep, user_request_slots_rep,
            empty_vec, empty_slots, empty_slots, empty_slots,
            turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep
        ])

        return goal_embedding

    def get_attention_weighted_goal(self, state):
        """
        Computes a weighted goal embedding using cosine similarity attention.
        """
        if not self.goal:
            return np.zeros_like(state)

        goal_words = [self.goal['intent']] + list(self.goal['inform_slots'].keys()) + list(self.goal['request_slots'].keys())
        vectors = [self.word2vec_model[word] for word in goal_words if word in self.word2vec_model]

        if not vectors:
            return np.zeros_like(state)

        goal_vec = np.mean(vectors, axis=0)  # میانگین بردارها
        goal_vec = self.resize_vector(goal_vec, state.shape[0])

        # محاسبه شباهت کسینوسی
        sim = np.dot(state, goal_vec) / (np.linalg.norm(state) * np.linalg.norm(goal_vec) + 1e-8)

        # وزن‌دار کردن بردار هدف
        weighted_goal = goal_vec * sim

        return weighted_goal

    def resize_vector(self, vec, new_size):
        """
        Resizes a vector to the target size using interpolation or truncation.
        """
        if len(vec) == new_size:
            return vec
        return np.interp(np.linspace(0, len(vec), new_size), np.arange(len(vec)), vec)
