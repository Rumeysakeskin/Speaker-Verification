import numpy as np
import time
from util import write_header, softmax, to_numpy, StandbyStatesHandler, CountdownTimer, Previous_Flow, \
    dummySerialTransfer
from states import OVEN_STATES, FLOW_STATES, COOKING_MODES
from pydub import AudioSegment
from pydub.playback import play
import threading
import copy

from enums import *


class OvenStateMachine(object):
    """
    State machine for controlling oven states.
    Currently only supports wakeword, stt and nlp states.
    """

    def __init__(self, config, listener, nemo_wakeword_inferencer,
                 quartznet_inferencer, nlp_engine, tts_inferencer, auto_correction_model, recipe_engine, logger,
                 serial_transfer, user_database, flow_states=FLOW_STATES):
        self.config = config
        self.listener = listener
        self.nemo_wakeword_inferencer = nemo_wakeword_inferencer
        self.quartznet_inferencer = quartznet_inferencer
        self.nlp_engine = nlp_engine
        self.tts_inferencer = tts_inferencer
        self.auto_correction_model = auto_correction_model
        self.recipe_engine = recipe_engine
        self.logger = logger
        self.serial_transfer = serial_transfer
        self.user_database = user_database

        self.printlog = logger.print
        self.flow_states = flow_states
        self.oven_states = dict(OVEN_STATES)
        self.cooking_modes = COOKING_MODES
        self.current_state = 0
        self.bytes_to_infer_wakeword = b""
        self.bytes_to_infer_stt = b""
        self.last_stt_output = ""
        self.last_nlp_output = ""
        self.tts_input = ""

        self.speech_speed = 2
        self.speech_volume = -1
        self.AUDIO_DEFAULT = -1

        self.last_complex_response = ""

        self.is_action_inferred = False
        self.is_flow_interrupted = False
        # this variable keeps extracted numeric values or cooking modes from commands if any
        self.extracted_nlp_value = -1

        # This flow is used for all inputs that expects simple commands such as: exit, start, no, yes, cancel
        self.simple_command_flow = False

        # this is used in complex flows that asks the user for a sepecific value
        self.complex_value_flow = False


        # this flow will start when a user wants to set the time/clock
        self.complex_clock_flow = False

        self.recipe_loop_flow = False

        self.recipe_ingredient_flow = False
        self.recipe_open_cooking_flow = False
        self.complex_close_cooking_before_recipe_flow = False

        self.incomplete_intent_flow = False
        self.complex_missing_entity_flow = False
        self.tutorial_flow = False

        self.rating_flow = False

        self.ingredient_flow = False
        self.user_registeration_flow = False

        self.provide_cooking_information_when_woken = False
        self.provide_preheat_information_when_woken = False

        self.needs_to_preheat = True

        # currently this is only used for rating flow
        # TODO: this will be applied to old flows in the future
        self.flow_counter = 0

        # this variable is used to count the number of times last system response is repeated
        self.repeat_last_response_counter = 0

        # this variable keeps the last requested command, this is used in complex scenarios
        self.current_action = ""
        self.action_number = -1

        # this variable is used to differentiate the cases where TTS live time inference is needed and the cases where responses are recorded
        # by default it's true, there is much fewer cases when live time TTS is needed
        self.tts_response_recorded = True

        self.listener.mode = self.flow_states[self.current_state]

        self.beep_sound_file = AudioSegment.from_wav("./files/IntelliOven_WakeUpBuzzer_Cropped.wav")

        self.cooking_timer = None
        self.alarm_timer = None
        self.preheating_timer = None

        self.standby_handler = StandbyStatesHandler()

        self.last_loop_is_standby = False

        self.last_detected_intent = None

        self.previous_flow = Previous_Flow(None, recipe_engine, None, -1)

        # This is used when cooking is open and we need to remember recipe info (queryType, value)
        self.last_recipe_inquiry = None

        self.is_recipe_cooking = False
        self.feedback_flow_start = False


    def reset_oven_states(self):
        self.oven_states = dict(OVEN_STATES)

    def switch_next_state(self):
        """
        Switches to next state.
        Adjusts listener mode.
        Resets variables.
        Currently for simple flows: wakeword --> stt --> nlp -->  --> command --> stt --> wakeword --> ...
        """
        self.reset_listener_variables()

        # TODO: these should be in WakeWord function perhaps
        if (self.is_flow_interrupted and self.current_state == 0 and (not self.is_action_inferred)):
            self.interrupted_flow_action()
            self.current_state = 4

        # elif self.feedback_flow_start:
        elif self.feedback_flow_start:

            self.current_action = "yemek tarifi geri bildirim"
            self.current_state = 4
            self.feedback_flow_start = False
            self.is_recipe_cooking = False


        elif self.current_state == 0 and self.provide_cooking_information_when_woken and self.get_oven_state("Cooking"):

            self.provide_cooking_information_when_woken = False
            self.current_action = "cooking information"
            self.current_state = 4

        elif self.current_state == 0 and self.provide_preheat_information_when_woken:

            self.provide_preheat_information_when_woken = False
            self.current_action = "preheat information"
            self.current_state = 4


        else:
            self.current_state = (self.current_state + 1) % (len(self.flow_states))
        self.listener.mode = self.flow_states[self.current_state]

    def switch_to_wakeword_state(self):
        self.current_state = 0
        self.listener.mode = self.flow_states[self.current_state]
        self.last_loop_is_standby = False
        self.stop_all_intent_flows()
        self.change_oven_state('OvenAwake', False)
        self.printlog("\nSwitching back to wake-word detection mode.\n\n\n\nListening for wake word...")

    def switch_to_wait_state(self):
        """
        Switches back to wait directly without going back into WakeWord detection.
        Adjusts listener mode.
        Resets variables.

        """
        self.bytes_to_infer_stt = b""
        self.listener.stt_frames = []
        self.listener.has_finished_speaking =  False

        self.current_state = self.flow_states.index("wait")
        self.listener.mode = self.flow_states[self.current_state]

    def switch_to_stt_state(self):
        """
        Switches back to STT directly without going back into WakeWord detection.
        Adjusts listener mode.
        Resets variables.

        """
        self.reset_listener_variables()
        self.current_state = self.flow_states.index("stt")
        self.listener.mode = self.flow_states[self.current_state]

    def reset_listener_variables(self):
        """
        Reset variables for audio capturing
        """
        # self.bytes_to_infer_stt = b""
        self.bytes_to_infer_wakeword = b""
        # self.listener.stt_frames = []
        self.listener.wake_up_frames = []
        # self.listener.has_finished_speaking =  False

    def get_current_state(self):
        """
        returns current state in string form
        """
        return self.flow_states[self.current_state]

    def reset_state_variables(self):
        '''
        reset all variables that keeps action, stt, nlp outputs.
        This will be called when we go back to wakeword detection mode
        '''

        self.extracted_nlp_value = -1
        self.current_action = ""
        self.action_number = -1
        self.last_nlp_output = self.nlp_engine.nlp_config.key_codes[self.action_number]
        self.tts_response_recorded = True
        self.stop_all_flows()
        self.recipe_engine.restart_recipe_results()
        self.flow_counter = 0
        self.repeat_last_response_counter = 0
        self.user_database.restart_active_data()

    def previous_state_variables(self):

        if self.current_action != 'yarım kalan akış':
            self.previous_flow._action_number = self.action_number
            self.previous_flow._action = self.current_action
            self.previous_flow._recipe = copy.deepcopy(self.recipe_engine)
            self.previous_flow._flow_control["complex_value_flow"] = self.complex_value_flow
            self.previous_flow._flow_control["simple_command_flow"] = self.simple_command_flow
            self.previous_flow._flow_control["complex_clock_flow"] = self.complex_clock_flow
            self.previous_flow._flow_control["recipe_loop"] = self.recipe_loop_flow
            self.previous_flow._flow_control["recipe_ingredient"] = self.recipe_ingredient_flow
            self.previous_flow._flow_control[
                'complex_close_cooking_before_recipe_flow'] = self.complex_close_cooking_before_recipe_flow

    # all flows  must be stopped here
    # intent flows are excluded
    def stop_all_flows(self):
        self.stop_complex_value_flow()
        self.stop_simple_command_flow()

        self.stop_complex_clock_flow()
        self.stop_recipe_loop()
        self.stop_recipe_ingredient()
        self.stop_complex_close_cooking_before_recipe_flow()
        self.stop_simple_command_flow()
        self.stop_rating_flow()
        self.stop_ingredient_flow()
        self.stop_user_registeration_flow()

    def stop_all_intent_flows(self):
        self.stop_tutorial_flow()
        self.stop_incomplete_intent_flow()
        self.stop_complex_missing_entity_flow()

    def any_complex_flow_active(self):
        return self.simple_command_flow or self.complex_value_flow or self.complex_clock_flow or self.recipe_loop_flow or self.recipe_ingredient_flow or self.complex_close_cooking_before_recipe_flow or self.simple_command_flow or self.rating_flow or self.ingredient_flow or self.user_registeration_flow

    def interrupted_flow_action(self):
        self.current_action = "yarım kalan akış"

    def process_wakeword(self):
        """
        Handles wakeword detection.
        Continuously makes inferences with audio data from listener class.
        """

        # reset variables
        if (not self.is_flow_interrupted):
            self.reset_state_variables()

        wake_word_bytes = b''
        # arbitrarily chosen inference chunk size limit
        # if current_bytes_chunk > self.config.inference_chunk_size_stt * 1.0:
        for chunk in self.listener.wake_up_frames:
            wake_word_bytes += chunk


        self.bytes_to_infer_wakeword = b""
        current_bytes_chunk = self.config.chunk_size * len(self.listener.wake_up_frames)

        # make an inference if we have enough audio data
        if current_bytes_chunk >= self.config.inference_chunk_size:
            # get part that will be extracted
            amount_frames = int(self.config.inference_chunk_size / self.config.chunk_size)

            # if this is the first time to do inference, we don't have any bytes in bytes_to_infer list
            if len(self.bytes_to_infer_wakeword) == 0:
                for i in range(0, amount_frames):
                    self.bytes_to_infer_wakeword += self.listener.wake_up_frames[i]

            # we already have some bytes, cut first bytes and append new bytes
            else:
                # cut first part
                self.bytes_to_infer_wakewordbytes_to_infer = self.bytes_to_infer_wakeword[self.config.stride:]
                # append new bytes
                for i in range(amount_frames - self.config.stride_chunk, amount_frames):
                    self.bytes_to_infer_wakeword += self.listener.wake_up_frames[i]

            # remove small portion of recorded audio, so model infers next audio section (with overlapping)
            del self.listener.wake_up_frames[0:self.config.stride_chunk]

            # make inference
            # Class 30 is HeyArçi
            positive_label_index = [30]
            logits = self.nemo_wakeword_inferencer.inference(self.bytes_to_infer_wakeword, wake_word_bytes)


            inference = int(np.argmax(logits))
            probs = softmax(logits)

            if inference in positive_label_index and probs[0][inference] < self.config.label_thresholds[inference]:
                inference = -1

            del logits

            if inference in positive_label_index:
                self.printlog(f"WakeWord Detected")
                self.logger.add_positive_count()
                self.logger.print_positive_count()

                self.change_oven_state('OvenAwake', True)
                # safeguard for memory leak
                del inference
                del positive_label_index

                play(self.beep_sound_file)
                self.printlog("\n\nListening for Speech to Text conversion...")

                # mark for stt, reset variables
                self.switch_next_state()

            # safeguard: increase strides if system is not fast enough to predict in time so it won't fall back
            if current_bytes_chunk > self.config.inference_chunk_size * 2:
                self.config.stride = self.config.stride_original * 2
                self.config.stride_chunk = self.config.stride_chunk_original * 2
            else:
                self.config.stride = self.config.stride_original
                self.config.stride_chunk = self.config.stride_chunk_original

            del current_bytes_chunk

        else:
            # wait for more audio
            time.sleep(self.config.audio_wait_time)

    def wait_for_speech(self):

        if self.listener.has_finished_speaking:
            self.switch_next_state()
            self.listener.stop_stt_listening()


        # close any complex scenarios if waiting period is over
        elif self.standby_handler.standby_period_ended():
            self.stop_all_intent_flows()
            self.standby_handler.delete_timer()
            self.reset_state_variables()
            self.switch_to_wakeword_state()
        # elif self.listener.has_finished_speaking == False

        # elif self.listener has not spoken for max duratıon


    def process_stt(self):
        """
        Handles speech-to-text.
        Makes inference with the data obtained from listener class.
        """
        self.is_action_inferred = False

        stt_bytes = b''
        # arbitrarily chosen inference chunk size limit
        # if current_bytes_chunk > self.config.inference_chunk_size_stt * 1.0:
        for chunk in self.listener.stt_frames:
            stt_bytes += chunk

        stt_bytes = write_header(stt_bytes, 1, 2, self.config.RATE)


        self.serial_transfer.archi_analyzing()

        self.printlog("\nPredicting audio...")
        now = time.time()

        # make prediction
        prediction = self.quartznet_inferencer.inference(stt_bytes)
        prediction = prediction[0]
        after = time.time()
        self.printlog(f"STT Prediction before correction:\n{prediction}")
        if self.config.override_stt_from_console:
            prediction = input('override stt: ')
            prediction = [(-1, prediction)]

        # make typo corrections to words (not numbers) with chars2vec
        if self.config.apply_auto_correction:
            prediction = self.auto_correction_model.apply_autocorrection(prediction)
            self.printlog(f"final STT Prediction:\n{prediction}")

        self.printlog(f"Time Elapsed: {after - now} seconds.")

        self.last_stt_output = prediction

        # Prediction is done, reset variables and switch to nlp state
        self.listener.clear_sst_frames()
        self.switch_next_state()
        self.printlog("\nSwitching to NLP")

        del prediction
        del stt_bytes



    def process_nlp(self):
        """
        Handles NLP.
        Makes inference with the output of STT module.
        """

        # throw any predictions less than 2 chars
        if (len(self.last_stt_output[0][1]) < 2):
            self.last_nlp_output = ""
            # we shouldn't restart if there's an ongoing complex flow with standby
            if (not self.standby_handler.in_standby_mode() and not self.any_complex_flow_active()):
                self.reset_state_variables()
            self.switch_next_state()
            return

        # should repeat last statement
        if self.process_nlp_repeat_last_response():
            self.repeat_last_response()
            self.current_state = 4
            return

        if self.any_complex_flow_active():

            # COMPLEX FLOW SPECIFIC NLP METHODS
            if (self.simple_command_flow):
                self.process_nlp_simple_commands()

            elif (self.user_registeration_flow):
                self.process_nlp_user_registeration()

            elif (self.ingredient_flow):
                self.process_nlp_ingredients()

            elif (self.complex_value_flow):
                self.process_nlp_value()


            elif (self.complex_clock_flow):
                self.process_nlp_clock()


            elif (self.recipe_open_cooking_flow):
                self.process_nlp_start_recipe_cooking()


            elif (self.recipe_loop_flow):
                self.process_nlp_recipe_loop_commands()


            elif (self.recipe_ingredient_flow):
                self.process_nlp_recipe_ingredient_commands()


            elif (self.complex_close_cooking_before_recipe_flow):
                # TODO different function
                self.process_nlp_confirmation()

            elif (self.rating_flow):
                self.process_nlp_extract_rating()



            if (self.last_nlp_output != ''):
                self.is_action_inferred = True
                self.is_flow_interrupted=False

            self.switch_next_state()
            return

        action_inferred = self.nlp_engine.inference_without_probability(self.last_stt_output)

        input_for_value_extraction = self.last_stt_output

        # extract value
        self.extracted_nlp_value = self.nlp_engine.extract_intent_specific_value_from_list(input_for_value_extraction,
                                                                                           action_inferred
                                                                                           )

        if action_inferred == 84 and self.extracted_nlp_value == -1:
            action_inferred = 82

        if action_inferred != -1:
            self.stop_tutorial_flow()

            if self.standby_handler.in_standby_mode():
                self.standby_handler.intent_detected()

        self.last_nlp_output = self.nlp_engine.nlp_config.key_codes[action_inferred]

        if (not (self.incomplete_intent_flow and action_inferred == -1) and not (
                self.complex_missing_entity_flow and action_inferred == -1)) or self.tutorial_flow:
            self.current_action = self.last_nlp_output

        # switch to next state
        self.switch_next_state()
        # self.printlog("\n\n\n\nProcessing speech-to-text...")
        self.action_number = action_inferred
        del action_inferred

    def repeat_last_response(self):
        if self.repeat_last_response_counter < 2:
            self.printlog("************************")
            self.printlog("REPEATING LAST RESPONSE")
            self.printlog("************************")

            self.tts_input = self.last_complex_response

            self.repeat_last_response_counter += 1
        else:
            self.stop_all_flows()
            self.standby_handler.intent_detected()
            self.tts_input = "command_unclear_closing_message"

    # Processes NLP in complex scenarios #
    # returns "pos", "neg" or "un-confirmed"
    def process_nlp_confirmation(self):
        self.last_nlp_output = self.nlp_engine.get_confirmation(self.last_stt_output)

    # Processes NLP in complex scenarios #
    # returns "pos", "neg" , "close", "open" ...etc or "un-confirmed"
    def process_nlp_simple_commands(self):
        self.last_nlp_output = self.nlp_engine.get_simple_commands(self.last_stt_output)

    def process_nlp_ingredients(self):
        self.last_nlp_output = self.nlp_engine.get_ingredients(self.last_stt_output)

    def process_nlp_value(self):

        input_for_value_extraction = self.last_stt_output
        self.extracted_nlp_value = self.nlp_engine.extract_intent_specific_value_from_list(input_for_value_extraction,
                                                                                           self.action_number)
        if (not (self.extracted_nlp_value == -1)):
            self.is_action_inferred = True
            self.is_flow_interrupted = False

    def process_nlp_program_confirmation(self):
        self.last_nlp_output = self.nlp_engine.confirm_program_selection(self.last_stt_output)

        # didn't get positive nor negative response, check for program selection
        if (self.last_nlp_output == "un-confirmed"):
            self.extracted_nlp_value = self.nlp_engine.extract_intent_specific_value_from_list(
                self.last_stt_output, self.action_number)

    def process_nlp_clock(self):
        self.last_nlp_output = self.nlp_engine.get_negative_response(self.last_stt_output)

        if (self.last_nlp_output != 'neg'):
            input_for_value_extraction = self.last_stt_output
            self.extracted_nlp_value = self.nlp_engine.extract_intent_specific_value_from_list(
                input_for_value_extraction,
                self.action_number)

    def process_nlp_recipe_ingredient_commands(self):

        self.last_nlp_output = self.nlp_engine.get_recipe_ingredient_command(self.last_stt_output)

    def process_nlp_user_registeration(self):
        self.last_nlp_output = self.nlp_engine.get_speaker_recognition_sentence(self.last_stt_output)

    def process_nlp_recipe_loop_commands(self):
        self.last_nlp_output = self.nlp_engine.get_recipe_loop_command(self.last_stt_output)

    def process_nlp_start_recipe_cooking(self):
        self.last_nlp_output = self.nlp_engine.get_start_response(self.last_stt_output)

    def process_nlp_extract_rating(self):
        self.last_nlp_output = self.nlp_engine.get_rating_values(self.last_stt_output)

    def process_nlp_repeat_last_response(self):
        return not self.nlp_engine.check_repeat_last_response_command(self.last_stt_output) == -1

    # this function uses actions matched with their keycodes in @nlp_config.py
    def command(self):
        self.printlog(
            f"Action Inferred: \"{self.last_nlp_output}\", KeyCode: {self.action_number}, ValueExtracted: {self.extracted_nlp_value}")

        if (self.current_action == "fırını kapat"):
            self.close_oven()
        elif (self.current_action == "wake word" or self.current_action == "fırını aç"):
            self.wake_word_sst()
        elif (self.current_action == "sıcaklık yükselt"):
            self.increase_or_decrease_temp('increase')
        elif (self.current_action == "sıcaklık düşür"):
            self.increase_or_decrease_temp('decrease')
        elif (self.current_action == "sıcaklık ayarlama isteği"):
            self.set_temp_intent()
        elif (self.current_action == "sıcaklığı belirli bir dereceye getir"):
            self.set_temp()
        elif (self.current_action == "maksimum sıcaklık bildirmek"):
            self.max_temp_inquiry()


        elif (self.current_action == "ses hızını yükselt"):
            self.increase_sound_speed()
        elif (self.current_action == "ses hızını düşür"):
            self.decrease_sound_speed()

        elif (self.current_action == "ses seviyesini yükselt"):
            self.increase_sound_volume()
        elif (self.current_action == "ses seviyesini düşür"):
            self.decrease_sound_volume()

        elif (self.current_action == "sesli konuşma özelliğini aç"):
            self.open_sound()

        elif (self.current_action == "sesli konuşma özelliğini kapat"):
            self.close_sound()

        elif (self.current_action == "ışık aç"):
            self.control_light("on")
        elif (self.current_action == "ışık kapat"):
            self.control_light("off")
        elif (self.current_action == "ışığı ayarlamak"):
            self.switch_light()

        elif (self.current_action == "bir önceki programa geç"):
            self.scroll_program("prev")
        elif (self.current_action == "bir sonraki programa geç"):
            self.scroll_program("next")
        elif (self.current_action == "pişirme modu seç"):
            self.set_program()
        elif (self.current_action == "pişirme modu tavsiyesi"):
            self.cooking_mode_recommendation()


        elif (self.current_action == "pişirmeyi aç"):
            self.control_program('open')
        elif (self.current_action == "pişirmeyi kapat"):
            self.control_program('close')

        elif (self.current_action == "cooking information"):
            self.provide_cooking_information()

        elif (self.current_action == "preheat information"):
            self.provide_preheating_information()

        elif (self.current_action == "saat yeni ayari al"):
            self.set_clock()
        elif (self.current_action == "saati belirli bir saate kur"):
            self.set_clock()
        elif (self.current_action == "saati bildirmek"):
            self.clock_inquiry()


        elif (self.current_action == "alarmı yeni ayari al"):
            self.set_alarm()
        elif (self.current_action == "alarmı belirli bir saate kur"):
            self.set_alarm()
        elif (self.current_action == "alarmı kapat"):
            self.modify_alarm("close")
        elif (self.current_action == "alarmı ileri saate kur"):
            self.modify_alarm("add")
        elif (self.current_action == "alarmı geri saate kur"):
            self.modify_alarm("subtract")
        elif (self.current_action == "alarm kalan sure bildirmek"):
            self.remaining_alarm_time_inquiry()


        elif (self.current_action == "pişirme süresi ayarla"):
            self.set_cooking_period()
        elif (self.current_action == "pişirme süresi iptal"):
            self.cancel_cooking_period()
        elif (self.current_action == "pişirme kalan süresi bildirmek"):
            self.cooking_period_inquiry()
        elif (self.current_action=="pişirme modu bilgi isteği"):
            self.cooking_mode_information()
        elif (self.current_action=="pişirme süresi arttır"):
            self.cooking_duration_increase_or_decrease("arttır")
        elif (self.current_action=="pişirme süresi azalt"):
            self.cooking_duration_increase_or_decrease("azalt")
        elif (self.current_action == "fan aç"):
            self.open_fan()
        elif (self.current_action == "fan kapat"):
            self.close_fan()


        elif (self.current_action == "yemek tarifi al"):
            self.get_recipe()


        elif (self.current_action == "closing cooking before recipe flow"):
            self.closing_cooking_before_recipe_flow()
        elif (self.current_action == "malzemeye göre yemek tarifi al"):
            self.recipe_flow_by_included_ingredients()
        elif (self.current_action == "bulunmayan malzemeye göre yemek tarifi al"):
            self.recipe_flow_by_excluded_ingredients()
        elif (self.current_action == 'pişirme süresine göre yemek tarifi al'):
            self.recipe_flow_by_time()
        elif (self.current_action == "çoklu seçime göre yemek tarifi al"):
            self.recipe_flow_by_multiple()
        elif (self.current_action == "tarif ismine göre yemek tarifi al"):
            self.recipe_flow_by_name()
        elif (self.current_action == "son kullanılan tarif"):
            self.last_active_recipe_enquiry()

        elif (self.current_action == "tarif loop"):
            self.recipe_loop()

        elif (self.current_action == "favori tarifi ekle"):
            self.add_favorite_recipe()

        elif (self.current_action == "favori tarifiler"):
            self.list_favorite_recipes()

        elif (self.current_action == "yemek tarifi geri bildirim"):
            self.last_recipe_feedback_flow()



        elif (self.current_action == "kullanıcıyı kaydet"):
            self.register_speaker()
        elif (self.current_action == "kullanıcının kaydını sil"):
            self.unregister_speaker()
        elif (self.current_action == "kişisel sağlık bilgiler hakkında bilgilendirme"):
            self.health_information_inquiry()

        elif (self.current_action == "register_speaker_go_to_health_info"):
            self.register_speaker_go_to_health_info()
        elif (self.current_action == "register_user_start_confirmation"):
            self.register_user_start_confirmation()
        elif (self.current_action == "health_information_modification"):
            self.health_information_modification()
        elif (self.current_action == "user_registeration_process"):
            self.user_registeration_process()
        elif (self.current_action == "add_user_ingredient"):
            self.add_user_ingredient()
        elif (self.current_action == "remove_user_ingredient"):
            self.remove_user_ingredient()

        elif (self.current_action == "confirm_unregistration"):
            self.confirm_unregistration()



        elif (self.current_action == "kapsamda olmayan niyet"):
            self.OOS_general()

        elif (self.current_action == "kapsamda olmayan diğer ürün"):
            self.OOS_product()

        elif (self.current_action == "kısmen anlaşılan niyet"):
            self.missing_action_flow()

        elif (self.current_action == "tam anlaşılamayan niyet"):
            self.missing_entity_flow()

        elif (self.current_action == "yarım kalan akış"):
            self.interrupted_flow()

        elif ((
                      self.last_nlp_output == "command_unclear" or self.current_action == "command_unclear" or self.current_action == "") and (
                      self.tutorial_flow or not self.any_complex_flow_active())):
            self.no_intent_detected()

        # here we handle simple scenarios where we directly output the response from key_codes @nlp_config.py
        else:
            if ("_" not in self.last_nlp_output):
                self.tts_input = 'command_unclear'

            else:
                self.tts_input = self.last_nlp_output

        # check if previously detected incompleted intent
        # must be in previous loops (not in current one)
        if self.incomplete_intent_flow and self.current_action != "tam anlaşılamayan niyet" and self.current_action != "kısmen anlaşılan niyet":
            self.incomplete_intent()

        self.switch_next_state()

    # oven states are only changed through this function
    def change_oven_state(self, key, value):

        # this is only for debug. Will be removed when serial transfer is finalized
        if isinstance(self.serial_transfer, dummySerialTransfer):
            self.oven_states[key] = value
            return

        if key == 'CookingDuration':
            if value == None:
                self.serial_transfer.unset_cooking_time()
            else:
                # TODO: should the cooking be unset first for increasing and decreasing cases
                mins, secs = divmod(value, 60)
                self.serial_transfer.set_cooking_time(mins, secs)

        elif key == 'Temp':
            self.serial_transfer.set_temperature(int(value / 5))

        elif key == 'LightsOn':
            if value:
                self.serial_transfer.lamp_control_on()
            else:
                self.serial_transfer.lamp_control_off()

        elif key == "CurrentProgram":
            self.serial_transfer.set_program(value)

        elif key == "OvenAwake":
            if value:
                self.serial_transfer.set_oven_on()
            else:
                self.serial_transfer.set_oven_off()

        elif key == "Cooking":
            if value:
                self.serial_transfer.start_cooking()
            else:
                self.serial_transfer.stop_cooking()

        else:
            self.printlog(f"DEBUG: State \"{key}\" not integrated to SerialTransfer (setter)")
            self.oven_states[key] = value

    def get_oven_state(self, key):

        # this is only for debug. Will be removed when serial transfer is finalized
        if isinstance(self.serial_transfer, dummySerialTransfer):
            return self.oven_states[key]

        if key == 'CurrentProgram':
            return self.serial_transfer.get_cooking_function()

        elif key == 'Temp':
            return 5 * self.serial_transfer.get_seted_temperature()

        elif key == 'Cooking':
            return bool(self.serial_transfer.get_cooking_status())

        elif key == 'LightsOn':
            return bool(self.serial_transfer.get_lamp_control())

        else:
            self.printlog(f"DEBUG: State \"{key}\"not integrated to SerialTransfer (getter)")
            return self.oven_states[key]

    def get_remaining_cooking_time_state_func(self):

        if isinstance(self.serial_transfer, dummySerialTransfer):
            now = time.time()
            cooking_time = now - self.get_oven_state('CookingStartingTime')
            cooking_duration_seconds = self.get_oven_state("CookingDuration") * 60

            if self.config.REDUCE_ALL_COOKING_DURATIONS_FOR_TESTING:
                cooking_duration_seconds = 60

            remaining_cooking_seconds = cooking_duration_seconds - cooking_time
            mins, secs = divmod(remaining_cooking_seconds, 60)
            mins, secs = int(mins), int(secs)


        else:
            remaining_cooking_seconds = self.serial_transfer.get_remaining_cooking_time()
            mins, secs = divmod(remaining_cooking_seconds, 60)

        return mins, secs

    def get_time_state_func(self):

        if isinstance(self.serial_transfer, dummySerialTransfer):

            hours, mins = self.get_oven_state('clock').get_clock()

        else:

            hours, mins = self.serial_transfer.get_time()

        return hours, mins

    def get_alarm_state_func(self):

        if isinstance(self.serial_transfer, dummySerialTransfer):

            now = time.time()
            alarm_time_passed = now - self.get_oven_state('AlarmStartingTime')
            alarm_value_seconds = self.get_oven_state("AlarmValue") * 60
            remaining_alarm_seconds = alarm_value_seconds - alarm_time_passed
            remaining_minutes, remaining_seconds = divmod(remaining_alarm_seconds, 60)
            remaining_minutes = int(remaining_minutes)

        else:

            hours, mins = self.serial_transfer.get_remaining_alarm()
            remaining_minutes = hours * 60 + mins

        return remaining_minutes

    def start_simple_command_flow(self):
        self.simple_command_flow = True

    def stop_simple_command_flow(self):
        self.simple_command_flow = False

    # complex flow is started when we ask the user a value
    def start_complex_value_flow(self):
        self.complex_value_flow = True

    # complex flow is ended when we receive the value from the user
    def stop_complex_value_flow(self):
        self.complex_value_flow = False


    def start_complex_clock_flow(self):
        self.complex_clock_flow = True

    def stop_complex_clock_flow(self):
        self.complex_clock_flow = False

    def start_recipe_loop(self):
        self.recipe_loop_flow = True

    def stop_recipe_loop(self):
        self.recipe_loop_flow = False

    def start_recipe_ingredient(self):
        self.recipe_ingredient_flow = True

    def stop_recipe_ingredient(self):
        self.recipe_ingredient_flow = False

    def start_recipe_open_cooking_flow(self):
        self.recipe_open_cooking_flow = True

    def stop_recipe_open_cooking_flow(self):
        self.recipe_open_cooking_flow = False

    def start_incomplete_intent_flow(self):
        self.incomplete_intent_flow = True

    def stop_incomplete_intent_flow(self):
        self.incomplete_intent_flow = False

    def start_complex_missing_entity_flow(self):
        self.complex_missing_entity_flow = True

    def stop_complex_missing_entity_flow(self):
        self.complex_missing_entity_flow = False

    def start_tutorial_flow(self):
        self.tutorial_flow = True

    def stop_tutorial_flow(self):
        self.tutorial_flow = False

    def start_complex_close_cooking_before_recipe_flow(self):
        self.complex_close_cooking_before_recipe_flow = True

    def stop_complex_close_cooking_before_recipe_flow(self):
        self.complex_close_cooking_before_recipe_flow = False

    def start_rating_flow(self):
        self.rating_flow = True

    def stop_rating_flow(self):
        self.rating_flow = False

    def start_ingredient_flow(self):
        self.ingredient_flow = True

    def stop_ingredient_flow(self):
        self.ingredient_flow = False

    def start_user_registeration_flow(self):
        self.user_registeration_flow = True

    def stop_user_registeration_flow(self):
        self.user_registeration_flow = False

    def remember_last_interrupt_response(self, response):
        self.last_complex_response = response

    def remember_last_complex_response(self):
        # if self.any_complex_flow_active():
        if self.tts_input != "command_unclear":
            self.last_complex_response = self.tts_input

    def process_tts(self):
        """
        Handles TTS using output from NLP engine.
        """

        self.remember_last_complex_response()

        if type(self.tts_input) is list:

            self.serial_transfer.archi_speaking()
            self.tts_inferencer.play_multi_file_tts_response(self.tts_input)
            self.previous_flow._tts_input = self.tts_input
        # "_" should only be used in static recordings
        elif "_" in self.tts_input:

            # skip TTS response if no intent was detected while in standby
            if (not ((self.standby_handler.in_standby_mode() and self.tts_input == "command_unclear") or (
                    self.last_loop_is_standby and self.tts_input == "command_unclear"))):
                if self.tts_input != "interrupted_flow_response":
                    self.previous_flow._tts_input = self.tts_input
                self.serial_transfer.archi_speaking()
                self.tts_inferencer.play_recording(self.tts_input)

            elif (not self.is_action_inferred and self.any_complex_flow_active()):
                #if self.current_action != 'yarım kalan akış'  and self.current_action != 'kapsamda olmayan niyet"' and self.current_action != 'kapsamda olmayan diğer ürün"' and self.current_action != 'kısmen anlaşılan niyet' and self.current_action != 'tam anlaşılamayan niyet' : #and self.current_action != 'kısmen anlaşılan niyet'
                if self.recipe_loop_flow:
                    self.is_flow_interrupted = True
                    self.previous_state_variables()

        # cases where there is a variable in the audio output

        elif self.tts_input != '':

            self.serial_transfer.archi_speaking()
            self.tts_inferencer.inference(self.tts_input, play_audio=True)
            self.previous_flow._tts_input = self.tts_input


        self.serial_transfer.archi_idle()

        if not self.any_complex_flow_active():
            self.reset_state_variables()

        # close any complex scenarios if waiting period is over
        if self.standby_handler.standby_period_ended():
            self.stop_all_intent_flows()
            self.standby_handler.delete_timer()
            self.reset_state_variables()

        # start waiting period if any
        self.standby_handler.start_waiting_period()

        # extract into list of complex flows
        if (self.any_complex_flow_active()):
            self.switch_to_wait_state()
            self.printlog("\nSwitching back to STT (Complex scenario)")

            if (self.standby_handler.in_standby_mode()):
                self.last_loop_is_standby = True

        elif (self.standby_handler.in_standby_mode()):
            self.last_loop_is_standby = True
            self.switch_to_wait_state()
            self.printlog("\nSwitching back to STT (Standby Mode)")
        else:

            self.switch_to_wakeword_state()

        ###
        self.extracted_nlp_value = -1
        ###

    #############################

    #### CODE FOR SCENARIOS ####

    #############################

    def close_oven(self):

        nlp_output = self.last_nlp_output

        # First phase of the scenario
        if (not self.simple_command_flow):

            # check if the oven is currently cooking
            # if True, this will be a complex scenario (we will ask the user for confirmation)
            if (self.get_oven_state('Cooking')):

                self.tts_input = "oven_close_while_cooking_confirm"

                self.start_simple_command_flow()
                self.standby_handler.short_standby()
            else:
                self.tts_input = "oven_close"  # oven_already_close eklenecek
                self.change_oven_state('OvenAwake', False)
                self.reset_oven_states()

        # Second phase of the complex scenario: we have asked for a confirmation and receieved some answer
        else:

            # the user has confirmed he wants to close the oven
            if (nlp_output == 'pos' or nlp_output == 'close'):
                # All we have to do is run the previous action
                self.tts_input = "oven_close"
                self.change_oven_state('OvenAwake', False)
                self.reset_oven_states()
                self.standby_handler.intent_detected()

                # switch the nlp output to the original command
                self.last_nlp_output = self.current_action
                self.stop_simple_command_flow()

                if self.cooking_timer:
                    self.cooking_timer.cancel()
                    del self.cooking_timer
                    self.cooking_timer = None

                if self.preheating_timer:
                    self.preheating_timer.cancel()
                    del self.preheating_timer
                    self.preheating_timer = None

            # The user has either provided a negative answer or did not provide a positive answer
            elif (nlp_output == 'neg' or nlp_output == 'dont_close'):
                self.tts_input = "cooking_continue"
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()
                self.stop_simple_command_flow()
            else:
                self.tts_input = "command_unclear"

    def wake_word_sst(self):

        self.tts_input = "oven_already_open"
        self.standby_handler.short_standby()

    def set_temp(self):
        '''
        a function that handles setting the temp to a specific X
        currently a complex scenario is not supported
        '''

        # the user should have provided the temp
        if (self.extracted_nlp_value != -1):

            new_temp = self.extracted_nlp_value

            # Check if temp is between valid range for current program
            if (new_temp > self.get_oven_state('CurrentProgramMinTemp') and new_temp <= self.get_oven_state(
                    'CurrentProgramMaxTemp')):
                self.change_oven_state('Temp', new_temp)
                self.tts_input = ['temp_set_1', f'{new_temp}', 'temp_set_2']
                self.printlog(f"Oven temperature has been set to {new_temp}\n")
                self.standby_handler.short_standby()


            else:
                self.tts_input = ["temperature_could_not_set", self.get_oven_state('CurrentProgramMinTemp'), "and", self.get_oven_state('CurrentProgramMaxTemp'), "between"]

                self.printlog(
                    f"Oven temperature outside current program range {self.get_oven_state('CurrentProgramMinTemp')} - {self.get_oven_state('CurrentProgramMaxTemp')}\n")
                self.standby_handler.short_standby()

    def set_temp_intent(self):

        if (not self.complex_value_flow):
            self.tts_input = "temp_set_intent"
            self.start_complex_value_flow()
            self.standby_handler.short_standby()

        else:
            if (self.extracted_nlp_value != -1):
                new_temp = self.extracted_nlp_value

                # Check if temp is between valid range for current program
                if (new_temp > self.get_oven_state('CurrentProgramMinTemp') and new_temp <= self.get_oven_state(
                        'CurrentProgramMaxTemp')):
                    self.change_oven_state('Temp', new_temp)

                    self.tts_input = ['temp_set_1', f'{new_temp}', 'temp_set_2']
                    self.printlog(f"Oven temperature has been set to {new_temp}\n")

                else:

                    self.tts_input = ["temperature_could_not_set", self.get_oven_state('CurrentProgramMinTemp'), "and",
                                      self.get_oven_state('CurrentProgramMaxTemp'), "between"]
                    self.printlog(
                        f"Oven temperature outside current program range {self.get_oven_state('CurrentProgramMinTemp')} - {self.get_oven_state('CurrentProgramMaxTemp')}\n")
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()
                self.stop_complex_value_flow()

            else:
                self.tts_input = "command_unclear"
                self.printlog(f"Did not receive a temperature value \n")

    def increase_sound_speed(self):
        if self.speech_speed == 1:
            self.speech_speed += 1
            self.tts_inferencer.change_speed(self.speech_speed)
            self.tts_input = "speech_speed_increased_to_2"

        elif self.speech_speed == 2:
            self.speech_speed += 1
            self.tts_inferencer.change_speed(self.speech_speed)
            self.tts_input = "speech_speed_increased_to_3"

        elif self.speech_speed == 3:
            self.tts_inferencer.change_speed(self.speech_speed)
            self.tts_input = "speech_speed_at_max"

    def decrease_sound_speed(self):
        if self.speech_speed == 3:
            self.speech_speed -= 1
            self.tts_inferencer.change_speed(self.speech_speed)
            self.tts_input = "speech_speed_decreased_to_2"

        elif self.speech_speed == 2:
            self.speech_speed -= 1
            self.tts_inferencer.change_speed(self.speech_speed)
            self.tts_input = "speech_speed_decreased_to_1"

        elif self.speech_speed == 1:
            self.tts_inferencer.change_speed(self.speech_speed)
            self.tts_input = "speech_speed_at_min"

    def increase_sound_volume(self):
        if self.speech_volume == 1:
            self.speech_volume += 1
            self.tts_inferencer.change_volume(self.speech_volume)
            self.tts_input = "speech_volume_increased_to_2"

        elif self.speech_volume == 2:
            self.speech_volume += 1
            self.tts_inferencer.change_volume(self.speech_volume)
            self.tts_input = "speech_volume_increased_to_3"

        elif self.speech_volume == 3:
            self.tts_inferencer.change_volume(self.speech_volume)
            self.tts_input = "speech_volume_at_max"

    def decrease_sound_volume(self):
        if self.speech_volume == 3:
            self.speech_volume -= 1
            self.tts_inferencer.change_volume(self.speech_volume)
            self.tts_input = "speech_volume_decreased_to_2"
        elif self.speech_volume == 2:
            self.speech_volume -= 1
            self.tts_inferencer.change_volume(self.speech_volume)
            self.tts_input = "speech_volume_decreased_to_1"

        elif self.speech_volume == 1:
            self.tts_inferencer.change_volume(self.speech_volume)
            self.tts_input = "speech_volume_at_min"

    def open_sound(self):
        self.tts_input = "speech_mode_open"

    def close_sound(self):
        self.tts_input = "speech_mode_close"
        self.tts_inferencer.close_speech()



    def increase_or_decrease_temp(self, mode):
        '''
        a function that handles both increase and decrease of temperature by X
        require parameter 'mode' indicates whether the user wants to increase or decrease
        '''

        if (self.extracted_nlp_value != -1):

            if (mode == "increase"):
                # arttirma
                new_temp = self.get_oven_state('Temp') + self.extracted_nlp_value
                mode_word = 'increased'

            elif (mode == "decrease"):
                # azaltma
                new_temp = self.get_oven_state('Temp') - self.extracted_nlp_value
                mode_word = 'decreased'

            # Check if temp is between valid range for current program
            if (new_temp > self.get_oven_state('CurrentProgramMinTemp') and new_temp <= self.get_oven_state(
                    'CurrentProgramMaxTemp')):
                self.change_oven_state('Temp', new_temp)

                # self.tts_input = f"Fırın sıcaklığını {new_temp}'a {mode_word}"
                self.tts_input = [f'temp_{mode_word}_1', f'{new_temp}', f'temp_{mode_word}_2']
                self.printlog(f"Oven temperature has been increased/decreased to {new_temp}\n")

                self.standby_handler.short_standby()

            else:

                self.tts_input = ["temperature_could_not_set", self.get_oven_state('CurrentProgramMinTemp'), "and",
                                  self.get_oven_state('CurrentProgramMaxTemp'), "between"]
                self.printlog(
                    f"Oven temperature outside current program range {self.get_oven_state('CurrentProgramMinTemp')} - {self.get_oven_state('CurrentProgramMaxTemp')}\n")
                self.standby_handler.short_standby()
        else:
            if (mode == "increase"):
                # arttirma
                new_temp = self.get_oven_state('Temp') + 10
                mode_word = 'increased'

            elif (mode == "decrease"):
                # azaltma
                new_temp = self.get_oven_state('Temp') - 10
                mode_word = 'decreased'

            # Check if temp is between valid range for current program
            if (new_temp > self.get_oven_state('CurrentProgramMinTemp') and new_temp <= self.get_oven_state(
                    'CurrentProgramMaxTemp')):
                self.change_oven_state('Temp', new_temp)
                # self.tts_input = f"Fırın sıcaklığını {new_temp}'a {mode_word}"
                self.tts_input = [f'temp_{mode_word}_1', f'{new_temp}', f'temp_{mode_word}_2']
                self.printlog(f"Oven temperature has been increased/decreased to {new_temp}\n")

                self.standby_handler.short_standby()

            else:

                self.tts_input = ["temperature_could_not_set", self.get_oven_state('CurrentProgramMinTemp'), "and",
                                  self.get_oven_state('CurrentProgramMaxTemp'), "between"]
                self.printlog(
                    f"Oven temperature outside current program range {self.get_oven_state('CurrentProgramMinTemp')} - {self.get_oven_state('CurrentProgramMaxTemp')}\n")
                self.standby_handler.short_standby()

    # This function gives tts output of program max temperature to user

    def max_temp_inquiry(self):
        program_name = self.cooking_modes[self.get_oven_state('CurrentProgram')]
        if (program_name == 'fanlı çalışma'):
            self.tts_input = 'program_max_temp_1'

        else:
            self.tts_input = f"program_max_temp_{self.get_oven_state('CurrentProgram') + 1}"
            self.standby_handler.short_standby()

    def control_light(self, mode):
        '''
        a function that handles both opening and closing the light
        require parameter 'mode' indicates whether the user wants to open or close the light

        '''

        mode_bool = mode == "on"

        # check the current status
        if (mode_bool != self.get_oven_state('LightsOn')):

            # we will change the state now
            self.change_oven_state('LightsOn', mode_bool)
            if mode == "on":
                self.tts_input = "light_open"
            else:
                self.tts_input = "light_close"

            self.printlog(f"Light status changed to:  {mode}\n")

        # nothing to change
        else:
            self.tts_input = "light_already_open" if mode == "on" else "light_already_close"
            self.printlog(f"Light is already:  {mode}\n")
            self.standby_handler.short_standby()

    def switch_light(self):
        if (self.get_oven_state('LightsOn')):
            self.control_light('off')
            self.standby_handler.short_standby()

        else:
            self.control_light('on')
            self.standby_handler.short_standby()

    def scroll_program(self, mode):
        if (mode == "next"):
            new_program = (self.get_oven_state('CurrentProgram') + 1) % (len(self.cooking_modes))
            print('new_program: ', new_program)
            self.tts_input = "selected_next_program"

        elif (mode == "prev"):
            new_program = (self.get_oven_state('CurrentProgram') - 1) % (len(self.cooking_modes))
            self.tts_input = "selected_prev_program"

        self.change_oven_state("CurrentProgram", new_program)
        self.printlog(f"changed program on display to: \"{self.cooking_modes[new_program]}\" \n")
        self.standby_handler.short_standby()

    def set_program(self):

        # check if the user has already made his choice
        if (self.extracted_nlp_value in self.cooking_modes):

            # if oven cooking is currently inactive
            if not self.get_oven_state('Cooking'):
                new_program = self.cooking_modes.index(self.extracted_nlp_value)
                self.change_oven_state("CurrentProgram", new_program)
                self.printlog(f"changed program on display to: \"{self.cooking_modes[new_program]}\" \n")
                self.tts_input = f"program_selected_{new_program + 1}"

            else:
                self.tts_input = "must_close_cooking"

            self.standby_handler.short_standby()


    def control_program(self, mode):
        '''
        a function that handles both opening and closing the cooking mode
        require parameter 'mode' indicates whether the user wants to open or close the light

        '''

        if mode == "open":
            if (self.get_oven_state('Temp') == 0):
                self.tts_input = "cooking_temp_not_set"
                self.standby_handler.short_standby()
            # TODO: must create default program (NONE)
            elif (self.get_oven_state('CurrentProgram') == 0):
                self.tts_input = "cooking_program_not_set"
                self.standby_handler.short_standby()
            elif (self.get_oven_state('CookingDuration') is None):
                self.tts_input = "cooking_duration_not_set"
                self.standby_handler.short_standby()

            elif (self.get_oven_state('Cooking')):
                self.tts_input = "must_close_cooking"
                self.standby_handler.short_standby()

            else:
                self.change_oven_state("LightsOn", True)
                self.change_oven_state("Cooking", True)

                self.printlog("WARNIING: USING DUMMY VALUE FOR PREHEATING: PREHEATING WILL BE FINISHED IN ONE MINUTE")
                if self.needs_to_preheat:
                    self.provide_preheat_information_when_woken = True
                    self.tts_input = ['starting_preheating_1', f"{self.get_oven_state('Temp')}",
                                      "starting_preheating_2"]

                    self.preheating_timer = threading.Timer(30, self.preheating_finished)
                    self.preheating_timer.start()
                else:
                    self.provide_cooking_information_when_woken = True
                    self.tts_input = "cooking_open"

                    self.printlog(
                        'Starting cooking timer for {} minutes.'.format(self.get_oven_state('CookingDuration')))

                    if isinstance(self.serial_transfer, dummySerialTransfer):
                        timer_duration_in_seconds = self.get_oven_state('CookingDuration') * 60
                        if self.config.REDUCE_ALL_COOKING_DURATIONS_FOR_TESTING:
                            timer_duration_in_seconds = 60
                        self.cooking_timer = threading.Timer(timer_duration_in_seconds, self.cooking_finished)
                        self.cooking_timer.start()
                        self.change_oven_state('CookingStartingTime', time.time())



        elif mode == "close":
            if (not self.simple_command_flow):
                if (self.get_oven_state('Cooking')):
                    self.tts_input = "cooking_confirm_close"
                    self.start_simple_command_flow()
                    self.standby_handler.short_standby()
                else:
                    self.tts_input = "cooking_was_not_started"

            else:

                if (
                        self.last_nlp_output == 'pos' or self.last_nlp_output == 'stop' or self.last_nlp_output == 'cancel'):
                    # All we have to do is run the previous action
                    self.change_oven_state("LightsOn", False)
                    self.tts_input = "cooking_close"
                    self.change_oven_state("Cooking", False)

                    if isinstance(self.serial_transfer, dummySerialTransfer):

                        if self.cooking_timer:
                            self.cooking_timer.cancel()
                            del self.cooking_timer
                            self.cooking_timer = None

                        if self.preheating_timer:
                            self.preheating_timer.cancel()
                            del self.preheating_timer
                            self.preheating_timer = None

                    # switch the nlp output to the original command
                    self.last_nlp_output = self.current_action
                    self.stop_simple_command_flow()

                # The user has either provided a negative answer or did not provide a positive answer
                elif (self.last_nlp_output == 'neg'):
                    self.tts_input = "cooking_continue"
                    self.stop_simple_command_flow()

                else:
                    self.tts_input = 'command_unclear'

    def set_clock(self):

        if (self.complex_clock_flow or self.extracted_nlp_value != -1):
            nlp_output = self.last_nlp_output
            if nlp_output == "neg":
                self.tts_input = "clock_exit"
                self.stop_complex_clock_flow()
                self.standby_handler.intent_detected()
            else:
                clock = self.extracted_nlp_value

                # todo: better condition
                if (type(clock) == str):

                    hours, mins = clock.split(":")[0], clock.split(":")[1]
                    if (self.oven_states['clock'].is_valid_time(int(hours), int(mins))):

                        self.tts_input = ['clock_set_1', f'{hours}', f'{mins}', 'clock_set_2']

                        self.printlog(f"clock set to {clock}")
                        self.oven_states['clock'].set_clock(int(hours), int(mins))
                        self.stop_complex_clock_flow()
                        self.standby_handler.intent_detected()

                    else:
                        self.tts_input = 'clock_value_invalid'
                        self.standby_handler.intent_detected()
                        self.standby_handler.short_standby()

                else:
                    self.tts_input = 'command_unclear'

        else:
            self.tts_input = 'clock_control_question'
            self.start_complex_clock_flow()
            self.standby_handler.short_standby()

    def set_alarm(self):

        if (self.complex_value_flow or self.extracted_nlp_value != -1):

            alarm_value = self.extracted_nlp_value

            if (alarm_value != -1):
                if (alarm_value[1] == 'clock'):
                    hours, mins = int(alarm_value[0].split(":")[0]), int(alarm_value[0].split(":")[1])
                    if (self.oven_states['clock'].is_valid_time(hours, mins)):

                        if (self.get_oven_state('AlarmValue') != None):  # if there is an alarm before the alarm starts, it will be cancelled it
                            self.alarm_timer.cancel()

                        mins = self.oven_states['clock'].time_difference(alarm_value[0])

                        self.tts_input = ['alarm_set_1', f'{mins}', 'alarm_set_2']
                        self.printlog(f"Alarm set to {mins} minutes")
                        self.change_oven_state('AlarmValue', mins)
                        timer_duration_in_seconds =  self.get_oven_state("AlarmValue") * 60
                        # timer starts to check remaining time for alarm
                        self.alarm_timer = threading.Timer(timer_duration_in_seconds, self.alarm_finished)
                        self.alarm_timer.start()
                        # self.oven_states['AlarmStartingTime'] = time.time()
                        self.change_oven_state('AlarmStartingTime', time.time())

                    else:
                        self.tts_input = "alarm_format_invalid"

                else:
                    if ( self.get_oven_state("AlarmValue") != None):
                        self.alarm_timer.cancel()
                    mins = alarm_value[0]
                    self.tts_input = ['alarm_set_1', f'{mins}', 'alarm_set_2']
                    self.printlog(f"Alarm set to {mins} minutes")
                    self.change_oven_state('AlarmValue', mins)
                    timer_duration_in_seconds = self.get_oven_state("AlarmValue") * 60

                    self.alarm_timer = threading.Timer(timer_duration_in_seconds, self.alarm_finished)
                    self.alarm_timer.start()
                    # self.oven_states['AlarmStartingTime'] = time.time()
                    self.change_oven_state('AlarmStartingTime', time.time())

                self.stop_complex_value_flow()
                self.standby_handler.intent_detected()
            else:
                self.tts_input = 'command_unclear'

        else:
            self.tts_input = 'alarm_control_question'
            self.start_complex_value_flow()
            self.standby_handler.short_standby()

    def modify_alarm(self, mode):

        if (self.get_oven_state('AlarmValue') is not None):
            if (mode == "close"):
                self.tts_input = "alarm_close"
                self.printlog("Alarm closed")
                self.change_oven_state('AlarmValue', None)
                self.alarm_timer.cancel()
                del self.alarm_timer
                self.alarm_timer = None

            elif (mode == "add" and self.extracted_nlp_value != -1):

                diff = int(self.extracted_nlp_value)
                self.tts_input = ['alarm_increased_1', f'{diff}', 'alarm_increased_2']
                self.printlog(f"{diff} minutes was added to alarm period")
                self.change_oven_state('AlarmValue', self.get_oven_state('AlarmValue') + diff)

            elif (mode == "subtract" and self.extracted_nlp_value != -1):

                diff = int(self.extracted_nlp_value)
                self.tts_input = ['alarm_reduced_1', f'{diff}', 'alarm_reduced_2']
                self.printlog(f"{diff} minutes was taken from alarm period")
                self.change_oven_state('AlarmValue', self.get_oven_state('AlarmValue') - diff)

            else:
                self.tts_input = f"alarm_value_unsure"

        else:
            self.tts_input = "alarm_not_set"
            self.printlog("Alarm is not open")

    # this function subtracts duration of alarm (15 mins alarm etc.) and the time that is passed in seconds
    # It returns remaining minutes
    def remaining_alarm_time_inquiry(self):
        alarm_value = self.get_oven_state('AlarmValue')

        if (alarm_value != None):

            now = time.time()
            alarm_time_passed = now - self.get_oven_state('AlarmStartingTime')
            alarm_value_seconds = self.get_oven_state("AlarmValue") * 60
            remaining_alarm_seconds = alarm_value_seconds - alarm_time_passed
            remaining_minutes, remaining_seconds = divmod(remaining_alarm_seconds, 60)
            remaining_minutes = int(remaining_minutes)
            self.tts_input = ['alarm_inquiry_1', f'{remaining_minutes}', 'alarm_inquiry_2']
        else:
            self.tts_input = "alarm_not_set"
            self.printlog("Alarm is not open")

    def clock_inquiry(self):
        hours, mins = self.get_time_state_func()
        self.tts_input = ['clock_inquiry_1', f'{hours}', f'{mins}']

    def set_cooking_period(self):

        if (self.complex_clock_flow or self.extracted_nlp_value != -1):

            if self.last_nlp_output == "neg":
                self.tts_input = "cooking_period_control_exit"
                self.stop_complex_clock_flow()
                self.standby_handler.intent_detected()
            else:
                cooking_period = self.extracted_nlp_value
                if (cooking_period != -1):
                    mins = cooking_period[0]
                    if (cooking_period[1] == 'clock'):
                        hours, mins = int(cooking_period[0].split(":")[0]), int(cooking_period[0].split(":")[1])
                        if (self.oven_states['clock'].is_valid_time(hours, mins)):
                            total_mins = self.oven_states['clock'].time_difference(cooking_period[0])
                            if total_mins < 360:

                                self.tts_input = ['cooking_duration_clock_set_1', f'{hours}', f'{mins}',
                                                  'cooking_duration_clock_set_2']
                                self.printlog(f"Cooking period set to {hours}:{mins} ")


                                self.change_oven_state('CookingDuration', total_mins)
                                self.stop_complex_clock_flow()
                                self.standby_handler.intent_detected()

                            else:
                                self.tts_input= "cooking_duration_max_six_hour"
                                self.standby_handler.intent_detected()
                                self.standby_handler.short_standby()

                        else:
                            self.tts_input = "cooking_period_value_invalid"
                            self.standby_handler.intent_detected()
                            self.standby_handler.short_standby()

                    else:
                        if mins < 360:

                            self.tts_input = ['cooking_duration_minute_set_1', f'{mins}', 'cooking_duration_minute_set_2']
                            self.printlog(f"Cooking period set to {mins} minutes")

                            self.change_oven_state('CookingDuration', mins)
                            self.stop_complex_clock_flow()
                            self.standby_handler.intent_detected()

                        else:

                            self.tts_input = "cooking_duration_max_six_hour"
                            self.standby_handler.intent_detected()
                            self.standby_handler.short_standby()

                else:
                    self.tts_input = 'command_unclear'


        else:
            self.tts_input = 'cooking_period_control_question'
            self.start_complex_clock_flow()
            self.standby_handler.short_standby()

    def cancel_cooking_period(self):
        if (not self.simple_command_flow):

            if (self.get_oven_state("CookingDuration") is None):
                self.tts_input = "cooking_period_already_set_question"
                self.start_simple_command_flow()
                self.standby_handler.short_standby()

            else:
                self.tts_input = "cooking_period_canceled"
                self.change_oven_state("CookingDuration", None)
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

        else:
            if (self.last_nlp_output == 'pos'):
                self.tts_input = 'cooking_period_control_question'

                # switch to cooking period control with the right action numbers
                self.current_action = "pişirme süresi ayarla"
                self.action_number = 100

                self.start_complex_clock_flow()
                self.stop_simple_command_flow()
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()
            # The user has provided a negative answer
            elif (self.last_nlp_output == 'neg'):
                self.tts_input = "cooking_period_control_exit"
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

            else:
                self.tts_input = 'command_unclear'

    def cooking_duration_increase_or_decrease(self, mode):

        if (self.extracted_nlp_value == -1):
            duration_value = 10
        else:
            duration_value = self.extracted_nlp_value

        if(not self.complex_value_flow):
            if (mode=="arttır"):
                if(not self.get_oven_state("Cooking")):
                    self.tts_input="cooking_duration_cooking_not_exist"

                else:
                    new_cooking_duration = self.get_oven_state('CookingDuration') + duration_value
                    if (new_cooking_duration<360):

                        self.change_oven_state('CookingDuration', new_cooking_duration)
                        self.tts_input = ["cooking_duration_increased_decreased_1", f"{duration_value}",
                                          "cooking_duration_increased_2", f"{self.get_oven_state('CookingDuration')}",
                                          "cooking_duration_increased_decreased_3"]
                        if (self.extracted_nlp_value == -1):
                            self.standby_handler.short_standby()
                    else:
                        if (self.extracted_nlp_value ==-1):
                            self.tts_input ="max_six_hour"
                        else:
                            self.tts_input = ["cooking_duration_extension_maximum_duration_1", f"{duration_value}",
                                              "cooking_duration_extension_maximum_duration_2"]

                        self.start_complex_value_flow()
                        self.standby_handler.short_standby()

            elif (mode=="azalt"):
                if(not self.get_oven_state("Cooking")):

                    self.tts_input="cooking_duration_cooking_not_exist"

                else:

                    new_cooking_duration = self.get_oven_state('CookingDuration') - duration_value
                    if (new_cooking_duration > 0):

                        self.change_oven_state('CookingDuration', new_cooking_duration)
                        self.tts_input = ["cooking_duration_increased_decreased_1", f"{duration_value}",
                                          "cooking_duration_decreased_2", f"{self.get_oven_state('CookingDuration')}",
                                          "cooking_duration_increased_decreased_3"]
                        if (self.extracted_nlp_value == -1):
                            self.standby_handler.short_standby()
                    else:
                        if (self.extracted_nlp_value == -1):
                            self.tts_input="less_than_ten_minutes"
                        else:
                            self.tts_input="cooking_duration_reduction_more_than_cooking_duration_1"

                        self.start_complex_value_flow()
                        self.standby_handler.short_standby()
        else:

            if (self.extracted_nlp_value != -1):

                self.standby_handler.intent_detected()
                self.stop_complex_value_flow()

                if (mode == "arttır"):
                    new_cooking_duration = self.get_oven_state('CookingDuration') + duration_value
                    if (new_cooking_duration<360):
                        self.change_oven_state('CookingDuration', new_cooking_duration)
                        self.tts_input = ["cooking_duration_increased_decreased_1", f"{duration_value}",
                                          "cooking_duration_increased_2", f"{self.oven_states['CookingDuration']}",
                                          "cooking_duration_increased_decreased_3"]

                    else:
                        self.tts_input = ["cooking_duration_extension_maximum_duration_final_1", f"{duration_value}",
                                          "cooking_duration_extension_maximum_duration_final_2"]

                elif (mode == "azalt"):
                    new_cooking_duration = self.get_oven_state('CookingDuration') - duration_value
                    if (new_cooking_duration > 0):
                        self.change_oven_state('CookingDuration', new_cooking_duration)
                        self.tts_input = ["cooking_duration_increased_decreased_1", f"{duration_value}",
                                          "cooking_duration_decreased_2", f"{self.get_oven_state('CookingDuration')}",
                                          "cooking_duration_increased_decreased_3"]
                    else:

                        self.tts_input = "cooking_duration_reduction_more_than_cooking_duration_2"

            else:
                    self.tts_input="command_unclear"


    def cooking_period_inquiry(self):
        if (not self.simple_command_flow):

            if (self.get_oven_state("CookingDuration") is None):
                # confusing response name "cooking_period_already_set_question"

                self.tts_input = "cooking_period_already_set_question"
                self.start_simple_command_flow()
                self.standby_handler.short_standby()

            else:
                if self.get_oven_state("Cooking"):

                    if not (self.get_oven_state('CookingStartingTime')== None):
                        now = time.time()
                        cooking_time = now - self.get_oven_state('CookingStartingTime')
                        cooking_duration_seconds = self.get_oven_state("CookingDuration") * 60
                        remaining_cooking_seconds = cooking_duration_seconds - cooking_time
                        mins, secs = divmod(remaining_cooking_seconds, 60)
                        mins, secs = int(mins), int(secs)
                        self.printlog("WARNING: Using dummy output for remaining cooking duration ")
                        self.tts_input = ['cooking_duration_inquiry_1', f'{mins}', 'cooking_duration_inquiry_2',
                                          f'{secs}',
                                          'cooking_duration_inquiry_3']

                else:
                    self.standby_handler.short_standby()
                    self.tts_input = "duration_set_but_cooking_not_started"


        else:
            if (self.last_nlp_output == 'pos'):
                self.tts_input = 'cooking_period_control_question'

                # switch to cooking period control with the right action numbers
                self.current_action = "pişirme süresi ayarla"
                self.action_number = 100

                self.start_complex_clock_flow()
                self.stop_simple_command_flow()

                self.standby_handler.short_standby()
            # The user has provided a negative answer
            elif (self.last_nlp_output == 'neg'):
                self.tts_input = "cooking_period_control_exit"
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

            else:
                self.tts_input = 'command_unclear'

    def cooking_mode_information(self):

        if (self.extracted_nlp_value == -1):
            program = self.get_oven_state('CurrentProgram')
            self.tts_input = [f"program_definition_{program + 1}", f"information_program_{program + 1}"]
        else:
            program = self.cooking_modes.index(self.extracted_nlp_value)
            self.tts_input = [f"program_definition_{program + 1}", f"information_program_{program + 1}"]

    def cooking_mode_recommendation(self):
        if (self.extracted_nlp_value == -1):
            self.tts_input = "suggestion_1"

        elif (self.extracted_nlp_value == "ızgara"):
            self.tts_input = "suggestion_2"

        self.standby_handler.short_standby()

    def voice_interrupt(self, response):
        self.tts_inferencer.play_when_available(response)
        self.reset_listener_variables()

    # this function will be called after cooking countdown/timer is done
    def cooking_finished(self):
        self.recipe_engine.load_last_active_recipe()
        if (self.is_recipe_cooking and not self.recipe_engine.last_recipe_has_been_rated_before()):
            self.feedback_flow_start = True
        self.change_oven_state('Cooking', False)
        self.change_oven_state('CookingDuration', None)
        self.printlog('Cooking Finished')

        self.remember_last_interrupt_response("cooking_finished")

        self.voice_interrupt('cooking_finished')

        self.provide_cooking_information_when_woken = False

    def preheating_finished(self):
        self.printlog("Preheating Finished. Starting cooking now...")

        self.remember_last_interrupt_response(
            ["preheating_finished_1", f"{self.get_oven_state('CookingDuration')}", "preheating_finished_2"])

        self.voice_interrupt(
            ["preheating_finished_1", f"{self.get_oven_state('CookingDuration')}", "preheating_finished_2"])
        self.provide_preheat_information_when_woken = False

        self.provide_cooking_information_when_woken = True

        if isinstance(self.serial_transfer, dummySerialTransfer):
            self.printlog('Starting cooking timer for {} minutes.'.format(self.get_oven_state('CookingDuration')))
            timer_duration_in_seconds = self.get_oven_state('CookingDuration') * 60
            if self.config.REDUCE_ALL_COOKING_DURATIONS_FOR_TESTING:
                timer_duration_in_seconds = 60
            self.cooking_timer = threading.Timer(timer_duration_in_seconds, self.cooking_finished)
            self.cooking_timer.start()
            self.change_oven_state('CookingStartingTime', time.time())

    # this function will be called after alarm countdown/timer is done
    # There is no TTS response since oven will have "beep" when alarm is finished
    def alarm_finished(self):

        self.change_oven_state('AlarmValue', None)
        self.printlog('Alarm Finished')

    def close_fan(self):
        nlp_output = self.last_nlp_output
        current_program = self.get_oven_state('CurrentProgram')
        # First phase of the scenario
        if (not self.simple_command_flow):

            # check if the oven is currently cooking
            # if True, this will be a complex scenario (we will ask the user for confirmation)
            if (self.get_oven_state('Cooking')):
                if not ((self.get_oven_state('CurrentProgram') == 3) or (self.get_oven_state('CurrentProgram') == 10)):

                    self.printlog(f"Current program : \"{self.cooking_modes[current_program]}\" \n")
                    self.tts_input = f"close_fan_program_{current_program + 1}"

                else:

                    self.printlog(f"Current program : \"{self.cooking_modes[current_program]}\" \n")
                    self.tts_input = f"close_fan_program_{current_program + 1}"
                    self.start_simple_command_flow()
                    self.standby_handler.short_standby()

            else:
                self.tts_input = 'cooking_was_not_started'


        else:
            # the user has confirmed he wants to close the fan
            if (nlp_output == 'pos'):

                self.tts_input = f"fan_close_confirm_close_{current_program + 1}"
                self.change_oven_state('FanOpen', False)

                if self.get_oven_state('CurrentProgram') == 3:
                    new_program = (self.get_oven_state('CurrentProgram') - 2) % (len(self.cooking_modes))
                    self.change_oven_state("CurrentProgram", new_program)
                    self.printlog(
                        f"Current program : \"{self.cooking_modes[self.get_oven_state('CurrentProgram')]}\" \n")

                elif self.get_oven_state('CurrentProgram') == 10:
                    new_program = (self.get_oven_state('CurrentProgram') - 1) % (len(self.cooking_modes))
                    self.change_oven_state("CurrentProgram", new_program)
                    self.printlog(
                        f"Current program : \"{self.cooking_modes[self.get_oven_state('CurrentProgram')]}\" \n")

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()


            elif (nlp_output == 'neg'):

                self.tts_input = "fan_confirm_not"
                self.printlog(f"Current program : \"{self.cooking_modes[self.get_oven_state('CurrentProgram')]}\" \n")

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

            else:
                self.tts_input = "command_unclear"

    def open_fan(self):
        nlp_output = self.last_nlp_output

        current_program = self.get_oven_state('CurrentProgram')
        # First phase of the scenario
        if (not self.simple_command_flow):

            # check if the oven is currently cooking
            # if True, this will be a complex scenario (we will ask the user for confirmation)

            if (self.get_oven_state('Cooking')):
                if not ((self.get_oven_state('CurrentProgram') == 1) or (self.get_oven_state('CurrentProgram') == 9)):

                    self.printlog(f"Current program : \"{self.cooking_modes[current_program]}\" \n")
                    self.tts_input = f"open_fan_program_{current_program + 1}"
                else:

                    self.printlog(f"Current program : \"{self.cooking_modes[current_program]}\" \n")
                    self.tts_input = f"open_fan_program_{current_program + 1}"
                    self.start_simple_command_flow()
                    self.standby_handler.short_standby()

            else:
                self.tts_input = 'cooking_was_not_started'

        else:

            if (nlp_output == 'pos'):

                self.tts_input = f"fan_open_confirm_open_{current_program + 1}"
                self.change_oven_state('FanOpen', True)

                if self.get_oven_state('CurrentProgram') == 1:
                    new_program = (self.get_oven_state('CurrentProgram') + 2) % (len(self.cooking_modes))
                    self.change_oven_state("CurrentProgram", new_program)
                    self.printlog(
                        f"Current program : \"{self.cooking_modes[self.get_oven_state('CurrentProgram')]}\" \n")

                elif self.get_oven_state('CurrentProgram') == 9:
                    new_program = (self.get_oven_state('CurrentProgram') + 1) % (len(self.cooking_modes))
                    self.change_oven_state("CurrentProgram", new_program)
                    self.printlog(
                        f"Current program : \"{self.cooking_modes[self.get_oven_state('CurrentProgram')]}\" \n")

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()


            elif (nlp_output == 'neg'):
                self.tts_input = "fan_confirm_not"
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

            else:
                self.tts_input = "command_unclear"

    def get_recipe(self):

        self.tts_input = 'recipes_list'
        self.standby_handler.short_standby()

    # parameter func will be the original recipe function this was called from
    def closing_cooking_before_recipe_flow(self, recipe_query_type=None):

        # first stage
        # this was called through recipe functions
        if recipe_query_type is not None:
            self.last_recipe_inquiry = (
            recipe_query_type, self.last_nlp_output, self.extracted_nlp_value, self.current_action)
            self.current_action = 'closing cooking before recipe flow'

            # start confirmation
            self.tts_input = 'must_close_cooking_before_recipe'
            self.start_complex_close_cooking_before_recipe_flow()
            self.standby_handler.short_standby()

        else:
            nlp_output = self.last_nlp_output

            # second stage
            if not self.simple_command_flow:
                if nlp_output == 'pos':

                    self.stop_complex_close_cooking_before_recipe_flow()
                    self.start_simple_command_flow()
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                    self.tts_input = 'confirm_close_cooking_before_recipe'

                elif nlp_output == 'neg':
                    self.stop_complex_close_cooking_before_recipe_flow()
                    self.standby_handler.intent_detected()

                    self.tts_input = 'closing_message'


                else:
                    self.tts_input = "command_unclear"

            # third stage
            else:
                if nlp_output == 'pos' or nlp_output == 'cancel':

                    self.stop_simple_command_flow()
                    self.standby_handler.intent_detected()

                    #########
                    self.change_oven_state("LightsOn", False)
                    self.change_oven_state("Cooking", False)
                    if self.cooking_timer:
                        self.cooking_timer.cancel()
                        del self.cooking_timer
                        self.cooking_timer = None

                    if self.preheating_timer:
                        self.preheating_timer.cancel()
                        del self.preheating_timer
                        self.preheating_timer = None
                    #########

                    #####
                    self.serial_transfer.archi_speaking()
                    # self.tts_inferencer.play_when_available('cooking_closed_searching_recipe')
                    self.voice_interrupt('cooking_closed_searching_recipe')
                    self.serial_transfer.archi_idle()
                    #####

                    recipe_query_type, self.last_nlp_output, self.extracted_nlp_value, self.current_action = self.last_recipe_inquiry
                    self.last_recipe_inquiry = None

                    # call the original func again
                    if recipe_query_type == 'included_ing':
                        self.recipe_flow_by_included_ingredients()
                    elif recipe_query_type == 'excluded_ing':
                        self.recipe_flow_by_excluded_ingredients()
                    elif recipe_query_type == 'duration':
                        self.recipe_flow_by_time()
                    elif recipe_query_type == 'multi_condition':
                        self.recipe_flow_by_multiple()
                    elif recipe_query_type == 'recipe_name':
                        self.recipe_flow_by_name()
                    elif recipe_query_type == "list_favorite":
                        self.list_favorite_recipes()





                elif nlp_output == 'neg':
                    self.stop_simple_command_flow()
                    self.standby_handler.intent_detected()

                    self.tts_input = 'closing_message'


                else:
                    self.tts_input = "command_unclear"

    # This function takes ingredients from user and check it is in recipes list or not.
    # Then if the ingredient is in recipe list, it decides the recipe that user wants.
    # if not, it will ask for new recipe or cancel the task according to user's answer.
    def recipe_flow_by_included_ingredients(self):
        # must close
        if (self.get_oven_state('Cooking')):
            self.closing_cooking_before_recipe_flow('included_ing')
            return

        nlp_output = self.last_nlp_output

        if (not self.recipe_ingredient_flow):
            if (self.extracted_nlp_value != -1):
                full_ingredient_sentece = " ".join(self.extracted_nlp_value)
                if (self.recipe_engine.search_by_included_ingredients(full_ingredient_sentece)):

                    self.tts_input = ['recipe_ingredient_control_1']
                    # adding all ingredients with "ve" keyword in between
                    for ingredient in self.extracted_nlp_value:
                        self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                    # remove the last "ve"
                    self.tts_input.pop(-1)

                    self.tts_input += ['recipe_ingredient_control_2',
                                       f'{self.recipe_engine.get_number_of_results()}',
                                       'recipe_ingredient_control_3', f'{self.recipe_engine.current_recipe.title}',
                                       'recipe_ingredient_control_4',
                                       f'{self.recipe_engine.current_recipe.duration}']

                    if (self.recipe_engine.get_number_of_results() == 1):

                        self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                    else:
                        self.tts_input += ['recipe_ingredient_control_5']

                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()
                else:
                    self.tts_input = ['recipe_is_not_found_1']
                    for ingredient in self.extracted_nlp_value:
                        self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                    self.tts_input.pop(-1)
                    self.tts_input += ['recipe_is_not_found_2']
                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()


        else:
            if self.recipe_engine.last_recipe_results != None:
                if (nlp_output == "start"):
                    self.standby_handler.intent_detected()
                    self.recipe_loop()
                    self.stop_recipe_ingredient()

                elif (nlp_output == "next_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                    not_last_recipe = self.recipe_engine.switch_to_next_recipe()

                    if not_last_recipe:
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}',
                                          'recipe_ingredient_control_5']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_last'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == "previous_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                    not_first_recipe = self.recipe_engine.switch_to_previous_recipe()

                    if not_first_recipe:
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}',
                                          'recipe_ingredient_control_5_previous']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_first'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == 'quit_recipe'):

                    self.tts_input = 'recipe_exit_confirm'
                    self.start_simple_command_flow()
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()


                elif (nlp_output == 'remove_from_favorite'):

                    if self.recipe_engine.recipe_exists_in_favorite(self.recipe_engine.current_recipe.json_path):
                        self.recipe_engine.remove_recipe_from_favorite(self.recipe_engine.current_recipe.json_path)
                        self.tts_input = "recipe_removed_from_favorite"
                    else:
                        self.tts_input = "recipe_is_not_in_favorite"

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()


                elif ((nlp_output == "previous_recipe" or nlp_output == "next_recipe") and (
                        self.recipe_engine.get_number_of_results() == 1)):
                    self.tts_input = "next_or_prev_but_only_one_recipe"
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()


                elif (self.simple_command_flow):

                    if (nlp_output == 'pos' or nlp_output == 'exit' or nlp_output == 'cancel'):

                        self.stop_recipe_ingredient()
                        self.stop_simple_command_flow()
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_exit']
                        self.standby_handler.intent_detected()
                        self.standby_handler.short_standby()


                    elif (nlp_output == 'neg'):

                        self.stop_simple_command_flow()
                        if self.recipe_engine.get_number_of_results() == 1:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2_singular_recipe']
                        else:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2']
                        self.standby_handler.intent_detected()
                        self.standby_handler.long_standby()

                    else:
                        self.tts_input = "command_unclear"

                else:
                    self.tts_input = "command_unclear"

            else:
                if (nlp_output == 'pos'):
                    self.standby_handler.intent_detected()
                    self.stop_recipe_ingredient()
                    self.tts_input = 'recipe_control_question'

                    self.standby_handler.long_standby()

                elif (nlp_output == 'neg'):

                    self.stop_recipe_ingredient()
                    self.standby_handler.intent_detected()
                    self.tts_input = ''

                else:

                    self.tts_input = "command_unclear"

    def recipe_flow_by_time(self):
        # must close
        if (self.get_oven_state('Cooking')):
            self.closing_cooking_before_recipe_flow('duration')
            return
        nlp_output = self.last_nlp_output

        if (not self.recipe_ingredient_flow):
            if (self.extracted_nlp_value != -1):
                if (self.recipe_engine.search_by_max_duration(self.extracted_nlp_value)):

                    self.tts_input = ['recipe_by_duration_control_1', f'{self.extracted_nlp_value}', 'recipe_by_duration_control_2',
                                      f'{self.recipe_engine.get_number_of_results()}',
                                      'recipe_by_duration_control_3', f'{self.recipe_engine.current_recipe.title}',
                                      'recipe_by_duration_control_4', f'{self.recipe_engine.current_recipe.duration}'
                                      ]

                    if (self.recipe_engine.get_number_of_results() == 1):

                        self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                    else:
                        self.tts_input += ['recipe_ingredient_control_5']

                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()
                else:
                    self.tts_input = ['recipe_by_duration_not_found_1', self.extracted_nlp_value,
                                      'recipe_by_duration_not_found_2']
                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()


            # could not extract numerical values
            else:
                self.tts_input = 'command_unclear'
                self.printlog('could not extract recipe time limit')

        else:
            if self.recipe_engine.last_recipe_results != None:
                if (nlp_output == "start"):
                    self.standby_handler.intent_detected()
                    self.recipe_loop()
                    self.stop_recipe_ingredient()

                elif (nlp_output == "next_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                    not_last_recipe = self.recipe_engine.switch_to_next_recipe()

                    if not_last_recipe:
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}',
                                          'recipe_ingredient_control_5']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_last'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == "previous_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                    not_first_recipe = self.recipe_engine.switch_to_previous_recipe()

                    if not_first_recipe:
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}',
                                          'recipe_ingredient_control_5_previous']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_first'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == 'quit_recipe'):

                    self.tts_input = 'recipe_exit_confirm'
                    self.standby_handler.intent_detected()
                    self.start_simple_command_flow()
                    self.standby_handler.short_standby()


                elif (nlp_output == 'remove_from_favorite'):

                    if self.recipe_engine.recipe_exists_in_favorite(self.recipe_engine.current_recipe.json_path):
                        self.recipe_engine.remove_recipe_from_favorite(self.recipe_engine.current_recipe.json_path)
                        self.tts_input = "recipe_removed_from_favorite"
                    else:
                        self.tts_input = "recipe_is_not_in_favorite"

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif ((nlp_output == "previous_recipe" or nlp_output == "next_recipe") and (
                        self.recipe_engine.get_number_of_results() == 1)):
                    self.tts_input = "next_or_prev_but_only_one_recipe"
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (self.simple_command_flow):

                    if (nlp_output == 'pos' or nlp_output == 'exit' or nlp_output == 'cancel'):

                        self.stop_recipe_ingredient()
                        self.stop_simple_command_flow()
                        # self.standby_handler.intent_detected()
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_exit']


                    elif (nlp_output == 'neg'):

                        self.stop_simple_command_flow()
                        if self.recipe_engine.get_number_of_results() == 1:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2_singular_recipe']
                        else:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2']
                        self.standby_handler.intent_detected()
                        self.standby_handler.long_standby()

                    else:
                        self.standby_handler.intent_detected()
                        self.tts_input = "command_unclear"
                else:
                    self.tts_input = "command_unclear"

            else:
                if (nlp_output == 'pos'):
                    self.standby_handler.intent_detected()
                    self.stop_recipe_ingredient()
                    self.tts_input = 'recipe_by_duration_return_question'

                    self.standby_handler.long_standby()

                elif (nlp_output == 'neg'):

                    self.stop_recipe_ingredient()
                    self.standby_handler.intent_detected()
                    self.tts_input = ''

                else:

                    self.tts_input = "command_unclear"

    def recipe_flow_by_excluded_ingredients(self):
        # must close
        if (self.get_oven_state('Cooking')):
            self.closing_cooking_before_recipe_flow('excluded_ing')
            return
        nlp_output = self.last_nlp_output

        if (not self.recipe_ingredient_flow):
            if (self.extracted_nlp_value != -1):
                full_ingredient_sentece = " ".join(self.extracted_nlp_value)
                if (self.recipe_engine.search_by_excluded_ingredients(full_ingredient_sentece)):

                    self.tts_input = ['recipe_ingredient_control_1']
                    # adding all ingredients with "ve" keyword in between
                    for ingredient in self.extracted_nlp_value:
                        self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                    # remove the last "ve"
                    self.tts_input.pop(-1)

                    self.tts_input += ['recipe_ingredient_control_2_excluded',
                                       f'{self.recipe_engine.get_number_of_results()}',
                                       'recipe_ingredient_control_3', f'{self.recipe_engine.current_recipe.title}',
                                       'recipe_ingredient_control_4',
                                       f'{self.recipe_engine.current_recipe.duration}']

                    if (self.recipe_engine.get_number_of_results() == 1):

                        self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                    else:
                        self.tts_input += ['recipe_ingredient_control_5']
                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()
                else:
                    # self.recipe_engine.last_recipe_results = None
                    self.tts_input = ['recipe_is_not_found_1']
                    for ingredient in self.extracted_nlp_value:
                        self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                    self.tts_input.pop(-1)
                    self.tts_input += ['recipe_is_not_found_excluded']
                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()



        else:
            if self.recipe_engine.last_recipe_results != None:
                if (nlp_output == "start"):
                    self.standby_handler.intent_detected()
                    self.recipe_loop()
                    self.stop_recipe_ingredient()

                elif (nlp_output == "next_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                    not_last_recipe = self.recipe_engine.switch_to_next_recipe()

                    if not_last_recipe:
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}',
                                          'recipe_ingredient_control_5']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_last'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == "previous_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                    not_first_recipe = self.recipe_engine.switch_to_previous_recipe()

                    if not_first_recipe:
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}',
                                          'recipe_ingredient_control_5_previous']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_first'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == 'quit_recipe'):

                    self.tts_input = 'recipe_exit_confirm'
                    self.standby_handler.intent_detected()
                    self.start_simple_command_flow()
                    self.standby_handler.short_standby()

                elif (nlp_output == 'remove_from_favorite'):

                    if self.recipe_engine.recipe_exists_in_favorite(self.recipe_engine.current_recipe.json_path):
                        self.recipe_engine.remove_recipe_from_favorite(self.recipe_engine.current_recipe.json_path)
                        self.tts_input = "recipe_removed_from_favorite"
                    else:
                        self.tts_input = "recipe_is_not_in_favorite"

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()


                elif ((nlp_output == "previous_recipe" or nlp_output == "next_recipe") and (
                        self.recipe_engine.get_number_of_results() == 1)):
                    self.tts_input = "next_or_prev_but_only_one_recipe"
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (self.simple_command_flow):

                    if (nlp_output == 'pos' or nlp_output == 'exit' or nlp_output == 'cancel'):

                        self.stop_recipe_ingredient()
                        self.stop_simple_command_flow()
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_exit']
                        self.standby_handler.intent_detected()
                        self.standby_handler.short_standby()

                    elif (nlp_output == 'neg'):

                        self.stop_simple_command_flow()
                        if self.recipe_engine.get_number_of_results() == 1:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2_singular_recipe']
                        else:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2']
                        self.standby_handler.intent_detected()
                        self.standby_handler.long_standby()

                    else:
                        self.tts_input = "command_unclear"
                else:
                    self.tts_input = "command_unclear"

            else:
                if (nlp_output == 'pos'):
                    self.standby_handler.intent_detected()
                    self.stop_recipe_ingredient()
                    self.tts_input = 'recipe_control_question'

                    self.standby_handler.long_standby()

                elif (nlp_output == 'neg'):

                    self.stop_recipe_ingredient()
                    self.standby_handler.intent_detected()
                    self.tts_input = ''

                else:

                    self.tts_input = "command_unclear"

    def recipe_flow_by_multiple(self):
        # must close
        if (self.get_oven_state('Cooking')):
            self.closing_cooking_before_recipe_flow('multi_condition')
            return
        nlp_output = self.last_nlp_output

        if (not self.recipe_ingredient_flow):

            if (self.extracted_nlp_value[0] != -1):
                extracted_value_list_0 = " ".join(self.extracted_nlp_value[0])
            else:
                extracted_value_list_0 = -1
            if (self.extracted_nlp_value[1] != -1):
                extracted_value_list_1 = " ".join(self.extracted_nlp_value[1])
            else:
                extracted_value_list_1 = -1

            if (self.extracted_nlp_value != -1):
                if (self.recipe_engine.search_by_included_and_excluded_ingredients_and_duration(extracted_value_list_0,
                                                                                                extracted_value_list_1,
                                                                                                self.extracted_nlp_value[
                                                                                                    2])):

                    self.tts_input = ['recipe_by_multiple_choices_1']

                    if (self.extracted_nlp_value[2] != -1):
                        self.tts_input += [f'{self.extracted_nlp_value[2]}', 'recipe_by_multiple_choices_duration_2']

                    if (self.extracted_nlp_value[0] != -1):

                        for ingredient in self.extracted_nlp_value[0]:
                            self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                            # remove the last "ve"
                        self.tts_input.pop(-1)

                        self.tts_input += ['recipe_by_multiple_choices_included_3']

                    if (self.extracted_nlp_value[1] != -1):

                        for ingredient in self.extracted_nlp_value[1]:
                            self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                            # remove the last "ve"
                        self.tts_input.pop(-1)

                        self.tts_input += ['recipe_by_multiple_choices_excluded_4']

                    self.tts_input += [f'{self.recipe_engine.get_number_of_results()}',
                                       'recipe_by_multiple_choices_5', f'{self.recipe_engine.current_recipe.title}',
                                       'recipe_by_multiple_choices_6', f'{self.recipe_engine.current_recipe.duration}']
                    if (self.recipe_engine.get_number_of_results() == 1):

                        self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                    else:
                        self.tts_input += ['recipe_ingredient_control_5']

                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()

                else:

                    self.tts_input = ['recipe_by_multiple_choices_not_found_1']

                    if (self.extracted_nlp_value[2] != -1):
                        self.tts_input += [f'{self.extracted_nlp_value[2]}',
                                           'recipe_by_multiple_choices_duration_not_found_2']

                    if (self.extracted_nlp_value[0] != -1):

                        for ingredient in self.extracted_nlp_value[0]:
                            self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                            # remove the last "ve"
                        self.tts_input.pop(-1)

                        self.tts_input += ['recipe_by_multiple_choices_included_not_found_3']

                    if (self.extracted_nlp_value[1] != -1):

                        for ingredient in self.extracted_nlp_value[1]:
                            self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']
                            # remove the last "ve"
                        self.tts_input.pop(-1)

                        self.tts_input += ['recipe_by_multiple_choices_excluded_not_found_4']

                    self.tts_input += ['recipe_by_multiple_choices_not_found_5']

                    self.start_recipe_ingredient()
                    self.standby_handler.short_standby()

        else:
            if self.recipe_engine.last_recipe_results != None:
                if (nlp_output == "start"):
                    self.standby_handler.intent_detected()
                    self.recipe_loop()
                    self.stop_recipe_ingredient()

                elif (nlp_output == "next_recipe"):

                    not_last_recipe = self.recipe_engine.switch_to_next_recipe()

                    if not_last_recipe :
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                      f'{self.recipe_engine.current_recipe.duration}', 'recipe_ingredient_control_5']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_last'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == "previous_recipe"):

                    not_first_recipe = self.recipe_engine.switch_to_previous_recipe()

                    if not_first_recipe :
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                      f'{self.recipe_engine.current_recipe.duration}']
                        if (self.recipe_engine.get_number_of_results() == 1):

                            self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                        else:
                            self.tts_input += ['recipe_ingredient_control_5_previous']
                    else:
                        self.tts_input = 'recipe_ingredient_control_already_first'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == 'quit_recipe'):

                    self.tts_input = 'recipe_exit_confirm'
                    self.standby_handler.intent_detected()
                    self.start_simple_command_flow()
                    self.standby_handler.short_standby()

                elif (nlp_output == 'remove_from_favorite'):

                    if self.recipe_engine.recipe_exists_in_favorite(self.recipe_engine.current_recipe.json_path):
                        self.recipe_engine.remove_recipe_from_favorite(self.recipe_engine.current_recipe.json_path)
                        self.tts_input = "recipe_removed_from_favorite"
                    else:
                        self.tts_input = "recipe_is_not_in_favorite"

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif ((nlp_output == "previous_recipe" or nlp_output == "next_recipe") and (
                        self.recipe_engine.get_number_of_results() == 1)):
                    self.tts_input = "next_or_prev_but_only_one_recipe"
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (self.simple_command_flow):

                    if (nlp_output == 'pos' or nlp_output == 'exit' or nlp_output == 'cancel'):

                        self.stop_recipe_ingredient()
                        self.stop_simple_command_flow()
                        self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_exit']
                        self.standby_handler.intent_detected()
                        self.standby_handler.short_standby()

                    elif (nlp_output == 'neg'):

                        self.stop_simple_command_flow()
                        if self.recipe_engine.get_number_of_results() == 1:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2_singular_recipe']
                        else:
                            self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                              'recipe_return_search_2']
                        self.standby_handler.intent_detected()
                        self.standby_handler.long_standby()

                    else:
                        self.tts_input = "command_unclear"
                else:
                    self.tts_input = "command_unclear"

            else:
                if (nlp_output == 'pos'):
                    self.standby_handler.intent_detected()
                    self.stop_recipe_ingredient()
                    self.tts_input = 'recipe_control_question'

                    self.standby_handler.long_standby()

                elif (nlp_output == 'neg'):

                    self.stop_recipe_ingredient()
                    self.standby_handler.intent_detected()
                    self.tts_input = ''

                else:
                    self.tts_input = "command_unclear"

    def recipe_flow_by_name(self):
        # must close
        if (self.get_oven_state('Cooking')):
            self.closing_cooking_before_recipe_flow('recipe_name')
            return
        nlp_output = self.last_nlp_output

        # first phase or third phase after confirmation
        if not self.recipe_ingredient_flow and not self.simple_command_flow:
            if self.extracted_nlp_value != -1:
                response = self.recipe_engine.search_by_recipe_title(self.extracted_nlp_value)

                # 1- Full match with recipe title
                if response == 1:

                    self.tts_input = ['recipe_by_name_found_1', f'{self.recipe_engine.get_number_of_results()}',
                                      'recipe_by_name_found_2', f'{self.recipe_engine.current_recipe.title}',
                                      'recipe_by_name_found_3', f'{self.recipe_engine.current_recipe.duration}']
                    if (self.recipe_engine.get_number_of_results() == 1):

                        self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                    else:
                        self.tts_input += ['recipe_ingredient_control_5']

                    self.standby_handler.short_standby()
                    self.start_recipe_ingredient()


                # 2.A- partial match with recipe title
                # 2.B- partial match with recipe ingredients
                elif response == 2:
                    self.tts_input = ['recipe_by_name_similar_1', f'{self.recipe_engine.get_number_of_results()}',
                                      'recipe_by_name_similar_2']

                    self.standby_handler.short_standby()
                    self.start_simple_command_flow()

                # no match
                elif response == -1:
                    self.tts_input = 'recipe_by_name_not_found'


        elif self.simple_command_flow and (not self.tts_input == "recipe_exit_confirm"):
            if (nlp_output == 'pos'):

                self.tts_input = [f'{self.recipe_engine.current_recipe.title}',
                                  'recipe_by_name_found_3', f'{self.recipe_engine.current_recipe.duration}']
                if (self.recipe_engine.get_number_of_results() == 1):

                    self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                else:
                    self.tts_input += ['recipe_ingredient_control_5']

                self.start_recipe_ingredient()
                self.stop_simple_command_flow()

                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()


            elif (nlp_output == 'neg'):

                self.stop_simple_command_flow()
                self.standby_handler.intent_detected()
                self.tts_input = 'closing_message'

            else:
                self.tts_input = 'command_unclear'

        else:
            if (nlp_output == "start"):
                self.standby_handler.intent_detected()
                self.recipe_loop()
                self.stop_recipe_ingredient()

            elif (nlp_output == "next_recipe"):

                    not_last_recipe = self.recipe_engine.switch_to_next_recipe()

                    if (self.recipe_engine.get_number_of_results() == 1):

                        self.tts_input = "next_or_prev_but_only_one_recipe"

                    else:

                        if not_last_recipe :
                            self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}']
                            if (self.recipe_engine.get_number_of_results() == 1):

                                self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                            else:
                                self.tts_input += ['recipe_ingredient_control_5']

                        else:

                                self.tts_input = 'recipe_ingredient_control_already_last'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

            elif (nlp_output == "previous_recipe"):

                    not_first_recipe = self.recipe_engine.switch_to_previous_recipe()

                    if (self.recipe_engine.get_number_of_results() == 1):

                        self.tts_input = "next_or_prev_but_only_one_recipe"

                    else:
                        if not_first_recipe :
                            self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_ingredient_control_4',
                                          f'{self.recipe_engine.current_recipe.duration}']
                            if (self.recipe_engine.get_number_of_results() == 1):

                                self.tts_input += ['recipe_ingredient_control_5_one_recipe']
                            else:
                                self.tts_input += ['recipe_ingredient_control_5_previous']
                        else:
                            self.tts_input = 'recipe_ingredient_control_already_first'

                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

            elif (nlp_output == 'quit_recipe'):

                self.tts_input = 'recipe_exit_confirm'
                self.standby_handler.intent_detected()
                self.start_simple_command_flow()
                self.standby_handler.short_standby()

            elif (nlp_output == 'remove_from_favorite'):

                if self.recipe_engine.recipe_exists_in_favorite(self.recipe_engine.current_recipe.json_path):
                    self.recipe_engine.remove_recipe_from_favorite(self.recipe_engine.current_recipe.json_path)
                    self.tts_input = "recipe_removed_from_favorite"
                else:
                    self.tts_input = "recipe_is_not_in_favorite"

                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

            elif ((nlp_output == "previous_recipe" or nlp_output == "next_recipe") and (
                    self.recipe_engine.get_number_of_results() == 1)):
                self.tts_input = "next_or_prev_but_only_one_recipe"
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

            elif (self.simple_command_flow):

                if (nlp_output == 'pos' or nlp_output == 'exit' or nlp_output == 'cancel'):

                    self.stop_recipe_ingredient()
                    self.stop_simple_command_flow()
                    self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_exit']
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                elif (nlp_output == 'neg'):

                    self.stop_simple_command_flow()
                    if self.recipe_engine.get_number_of_results() == 1:
                        self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                          'recipe_return_search_2_singular_recipe']
                    else:
                        self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                          'recipe_return_search_2']
                    self.standby_handler.intent_detected()
                    self.standby_handler.long_standby()

                else:
                    self.tts_input = "command_unclear"
            else:
                self.tts_input = "command_unclear"

    def last_active_recipe_enquiry(self):

        if not self.simple_command_flow:

            # if loading was succesfull
            if self.recipe_engine.load_last_active_recipe():
                self.tts_input = ['last_used_recipe_1', self.recipe_engine.current_recipe.get_title_recording(),
                                  'last_used_recipe_2', self.recipe_engine.current_recipe.duration,
                                  'last_used_recipe_3']
                self.start_simple_command_flow()
                self.standby_handler.short_standby()

            # could not load past recipe
            else:
                self.tts_input = "last_used_recipe_not_found"

        else:
            nlp_output = self.last_nlp_output

            if nlp_output == 'pos' or nlp_output == 'start':
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()
                self.recipe_loop()

            elif nlp_output == 'neg' or nlp_output == 'cancel' or nlp_output == 'close':
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()
                self.tts_input = 'closing_message'

            else:
                self.tts_input = 'command_unclear'

    # This function will first be called from inside another command function
    # Then this function will be be called from the general command function for all other times
    def recipe_loop(self):

        if not self.recipe_loop_flow:
            self.tts_input = 'recipe_start_message'
            self.current_action = 'tarif loop'
            self.start_recipe_loop()
            self.standby_handler.long_standby()


        else:
            nlp_output = self.last_nlp_output

            # second stage: confirm exit of recipe loop
            if self.simple_command_flow:

                if (nlp_output == 'pos' or nlp_output == 'exit' or nlp_output == 'cancel'):
                    self.tts_input = [self.recipe_engine.current_recipe.get_title_recording(), 'recipe_exit']
                    self.standby_handler.intent_detected()
                    self.stop_simple_command_flow()
                    self.stop_recipe_loop()



                elif (nlp_output == 'neg'):
                    self.tts_input = ['recipe_return_1', self.recipe_engine.current_recipe.get_current_step_recording(),
                                      'recipe_return_2']

                    self.recipe_engine.current_recipe.remove_ingredients_step_if_exists()

                    self.standby_handler.intent_detected()
                    self.stop_simple_command_flow()
                    self.standby_handler.long_standby()


                else:
                    self.tts_input = "command_unclear"

            # second stage: confirm starting suggested cooking program
            elif self.recipe_open_cooking_flow:
                if (nlp_output == 'start'):
                    self.is_recipe_cooking = True
                    self.tts_input = 'recipe_cooking_start'

                    # self.printlog("Warning: using dummy value for suggested cooking program and temp", )
                    # self.change_oven_state('CookingDuration', self.recipe_engine.current_recipe.duration)
                    # self.change_oven_state('Temp', self.get_oven_state('DummySuggestedTemp'))
                    # self.serial_transfer.set_temperature(int(self.get_oven_state('DummySuggestedTemp') / 5))
                    # self.change_oven_state('CurrentProgram', self.get_oven_state('DummySuggestedProgram'))

                    self.change_oven_state("LightsOn", True)
                    self.change_oven_state("Cooking", True)
                    self.change_oven_state('CookingDuration', self.recipe_engine.current_recipe.duration)

                    self.printlog(
                        "WARNIING: USING DUMMY VALUE FOR PREHEATING: PREHEATING WILL BE FINISHED IN ONE MINUTE")

                    if self.needs_to_preheat:
                        self.provide_preheat_information_when_woken = True
                        self.printlog(
                            "WARNIING: USING DUMMY VALUE FOR TEMP: 180")

                        self.tts_input = ['starting_preheating_1', f"180",
                                          "starting_preheating_2"]

                        self.preheating_timer = threading.Timer(30, self.preheating_finished)
                        self.preheating_timer.start()
                    else:

                        self.provide_cooking_information_when_woken = True
                        self.tts_input = "cooking_open"

                        self.printlog(
                            'Starting cooking timer for {} minutes.'.format(self.get_oven_state('CookingDuration')))

                        if isinstance(self.serial_transfer, dummySerialTransfer):
                            timer_duration_in_seconds = self.get_oven_state('CookingDuration') * 60
                            self.cooking_timer = threading.Timer(timer_duration_in_seconds, self.cooking_finished)

                            self.cooking_timer.start()
                            self.change_oven_state('CookingStartingTime', time.time())

                    # self.printlog(
                    #     'Starting cooking timer for {} minutes.'.format(self.recipe_engine.current_recipe.duration))
                    # timer_duration_in_seconds = self.recipe_engine.current_recipe.duration * 60
                    # self.change_oven_state('CookingStartingTime', time.time())

                    # self.cooking_timer = threading.Timer(timer_duration_in_seconds, self.cooking_finished)
                    # self.cooking_timer.start()

                    self.standby_handler.intent_detected()
                    self.stop_recipe_open_cooking_flow()
                    self.stop_recipe_loop()

                    self.recipe_engine.save_current_recipe()

                    # this flow will check if recipe is added to favorite if not will ask the user for adding
                    self.add_favorite_recipe()


                elif (nlp_output == 'exit'):
                    self.stop_recipe_open_cooking_flow()

                    self.start_simple_command_flow()
                    self.tts_input = 'recipe_exit_confirm'
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()

                else:
                    self.tts_input = "command_unclear"
            # 4 commands loop
            else:

                if nlp_output == 'next_step':

                    next_step = self.recipe_engine.current_recipe.next_step()

                    # have reached the last step
                    if next_step is None:

                        self.recipe_engine.current_recipe.prev_step()

                        self.tts_input = 'recipe_confirm_start_cooking'

                        self.start_recipe_open_cooking_flow()

                    # this function will return false if we reached the last step, true otherwise
                    else:
                        self.tts_input = next_step

                    self.standby_handler.intent_detected()
                    self.standby_handler.long_standby()




                elif nlp_output == 'prev_step':

                    prev_step = self.recipe_engine.current_recipe.prev_step()

                    # have reached the last step
                    if prev_step is None:
                        self.tts_input = 'recipe_already_at_first_step'

                    # this function will return false if we reached the last step, true otherwise
                    else:
                        self.tts_input = prev_step

                    self.standby_handler.intent_detected()
                    self.standby_handler.long_standby()


                elif nlp_output == 'ingredients':
                    self.tts_input = self.recipe_engine.current_recipe.get_ingredients_recording()
                    self.standby_handler.intent_detected()
                    self.standby_handler.long_standby()


                elif nlp_output == 'repeat':
                    self.tts_input = self.recipe_engine.current_recipe.get_current_step_recording()
                    self.standby_handler.intent_detected()
                    self.standby_handler.long_standby()


                elif nlp_output == 'exit':

                    self.start_simple_command_flow()
                    self.tts_input = 'recipe_exit_confirm'
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()


                # TODO:
                # empty input < 3 chars
                # tts_input= "empty_input"
                elif nlp_output == '':
                    self.tts_input = "command_unclear"

                # Un-confirmed
                else:
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()
                    self.tts_input = "recipe_command_unclear"

    def add_favorite_recipe(self):
        # first stage
        if not self.simple_command_flow:

            # was not found in favorite
            if not self.recipe_engine.recipe_exists_in_favorite(self.recipe_engine.current_recipe.json_path):
                if self.needs_to_preheat:
                    self.tts_input = ['starting_preheating_1', f"180",
                                      "starting_preheating_2", 'add_recipe_question']
                else:
                    self.tts_input = 'cooking_started_add_recipe_question'
                self.start_simple_command_flow()
                self.standby_handler.short_standby()
                self.current_action = "favori tarifi ekle"


        # second stage: get confirmation
        else:

            nlp_output = self.last_nlp_output

            if nlp_output == "pos" or nlp_output == "add":

                self.recipe_engine.add_current_recipe_to_favorite()

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

                self.tts_input = "recipe_added"


            elif nlp_output == "neg" or nlp_output == "dont_add":

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

                self.tts_input = "closing_message"

            else:
                self.tts_input = "command_unclear"

    def list_favorite_recipes(self):

        # must close
        if (self.get_oven_state('Cooking')):
            self.closing_cooking_before_recipe_flow('list_favorite')
            return

        # first step
        if not self.recipe_ingredient_flow:

            if self.recipe_engine.load_favorite_recipes():
                self.recipe_engine.loop_favorites()
                self.tts_input = ['favorite_recipes_list_1', f"{self.recipe_engine.get_number_of_results()}",
                                  'favorite_recipes_list_2', "favorite_recipe_keyword",
                                  f"{self.recipe_engine.current_recipe.title}"]

                if self.recipe_engine.get_number_of_results() == 1:
                    self.tts_input.append("favorite_recipes_list_3_singular_recipe")
                else:
                    self.tts_input.append("favorite_recipes_list_3")

                self.standby_handler.short_standby()
                self.start_recipe_ingredient()

            else:
                self.tts_input = "no_favorite_recipes_availble"

        # favorite loop
        else:

            nlp_output = self.last_nlp_output

            if (nlp_output == "start"):
                self.standby_handler.intent_detected()
                self.recipe_loop()
                self.stop_recipe_ingredient()

            elif (nlp_output == "next_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                not_last_recipe = self.recipe_engine.switch_to_next_recipe()

                if not_last_recipe:
                    self.tts_input = ["favorite_recipe_keyword", f"{self.recipe_engine.current_recipe.title}",
                                      'favorite_recipes_list_3']
                else:
                    self.tts_input = 'recipe_ingredient_control_already_last'

                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

            elif (nlp_output == "previous_recipe" and not (self.recipe_engine.get_number_of_results() == 1)):

                not_first_recipe = self.recipe_engine.switch_to_previous_recipe()

                if not_first_recipe:
                    self.tts_input = ["favorite_recipe_keyword", f"{self.recipe_engine.current_recipe.title}",
                                      'favorite_recipes_list_3']
                else:
                    self.tts_input = 'recipe_ingredient_control_already_first'

                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()


            # Before adding this flow here, the flow must cover the case for a singular favorite recipe when removed from favorite list
            elif (nlp_output == 'remove_from_favorite'):

                if self.recipe_engine.recipe_exists_in_favorite(self.recipe_engine.current_recipe.json_path):
                    self.recipe_engine.remove_recipe_from_favorite(self.recipe_engine.current_recipe.json_path)
                    self.tts_input = "recipe_removed_from_favorite"
                else:
                    self.tts_input = "recipe_is_not_in_favorite"

                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

            elif (nlp_output == 'quit_recipe'):

                self.tts_input = 'recipe_exit_confirm'
                self.start_simple_command_flow()
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()




            elif ((nlp_output == "previous_recipe" or nlp_output == "next_recipe") and (
                    self.recipe_engine.get_number_of_results() == 1)):
                self.tts_input = "next_or_prev_but_only_one_favorite_recipe"
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()



            elif (self.simple_command_flow):

                if (nlp_output == 'pos' or nlp_output == 'exit' or nlp_output == 'cancel'):

                    self.stop_recipe_ingredient()
                    self.stop_simple_command_flow()
                    self.tts_input = [f'{self.recipe_engine.current_recipe.title}', 'recipe_exit']
                    self.standby_handler.intent_detected()
                    self.standby_handler.short_standby()


                elif (nlp_output == 'neg'):

                    self.stop_simple_command_flow()

                    if self.recipe_engine.get_number_of_results() == 1:
                        self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                          'recipe_return_search_2_singular_recipe']
                    else:
                        self.tts_input = ['recipe_return_search_1', f'{self.recipe_engine.current_recipe.title}',
                                          'recipe_return_search_2']

                    self.standby_handler.intent_detected()
                    self.standby_handler.long_standby()

                else:
                    self.tts_input = "command_unclear"

            else:
                self.tts_input = "command_unclear"

    def OOS_general(self):
        self.tts_input = ['out_of_scope_general_response_1', self.extracted_nlp_value,
                          'out_of_scope_general_response_2']

    def OOS_product(self):
        self.tts_input = ['out_of_scope_product_response_1', self.extracted_nlp_value,
                          'out_of_scope_product_response_2']

    def missing_entity_flow(self):
        # third phase (last attempt for the user)
        if self.incomplete_intent_flow:
            self.incomplete_intent()
            return

        # first phase
        if not self.complex_missing_entity_flow:
            extracted_nlp_value = self.extracted_nlp_value

            self.tts_input = ['incomplete_intent_response_1']

            if extracted_nlp_value == 'duration':
                if self.get_oven_state('Cooking'):
                    self.tts_input.append('duration_intent_message_2')
                else:
                    self.tts_input.append('duration_intent_message_1')

            elif extracted_nlp_value == 'finish':
                if self.get_oven_state('Cooking'):
                    self.tts_input.append('finish_intent_message_1')
                else:
                    self.tts_input.append('finish_intent_message_2')

            elif extracted_nlp_value == 'time':
                self.tts_input.append('time_intent_message')

            elif extracted_nlp_value == 'remaining_time':

                if self.get_oven_state('Cooking') or self.get_oven_state('AlarmValue') is not None:
                    if self.get_oven_state('Cooking'):
                        self.tts_input.append('remaining_time_intent_message_cooking')

                    if self.get_oven_state('AlarmValue') is not None:
                        self.tts_input.append('remaining_time_intent_message_alarm')

                    self.tts_input.append('you_can_say_keyword')

                else:
                    self.tts_input = 'remaining_time_intent_message_not_active'


            elif extracted_nlp_value == 'close':

                self.tts_input.append('close_intent_message_oven')

                if self.get_oven_state('LightsOn'):
                    self.tts_input.append('close_intent_message_light')

                if self.get_oven_state('Cooking'):
                    self.tts_input.append('close_intent_message_cooking')

                self.tts_input.append('you_can_say_keyword')

            elif extracted_nlp_value == 'change':
                self.tts_input = ['incomplete_intent_response_4']
                self.tts_input.append('change_intent_message')

            elif extracted_nlp_value == 'start':
                self.tts_input.append('start_intent_message')
            elif extracted_nlp_value == 'open':
                self.tts_input.append('open_intent_message')
            elif extracted_nlp_value == 'set':
                self.tts_input = ['incomplete_intent_response_4']
                self.tts_input.append('set_intent_message')

            elif extracted_nlp_value == 'alarm':
                if self.get_oven_state('AlarmValue') is not None:
                    self.tts_input.append('alarm_intent_message_1')
                else:
                    self.tts_input.append('alarm_intent_message_2')

            self.standby_handler.short_standby()
            self.start_complex_missing_entity_flow()
        # second phase
        else:

            if not self.incomplete_intent_flow:
                self.stop_complex_missing_entity_flow()
                self.standby_handler.intent_detected()

                self.tts_input = 'missing_entity_response'
                self.start_incomplete_intent_flow()
                self.standby_handler.short_standby()

    def missing_action_flow(self):

        # third phase (last attempt for the user)
        if self.incomplete_intent_flow:
            self.incomplete_intent()
            return
        # first phase
        if not self.simple_command_flow:
            extracted_nlp_value = self.extracted_nlp_value
            # remember this for the confirmation phase
            self.last_detected_intent = extracted_nlp_value

            self.tts_input = ['incomplete_intent_response_2']

            if extracted_nlp_value == 'temp':
                self.tts_input.append('temp_intent_question')
            elif extracted_nlp_value == 'fan':
                self.tts_input.append('fan_intent_question')
            elif extracted_nlp_value == 'cooking-mode':
                self.tts_input.append('cooking_mode_intent_question')
            elif extracted_nlp_value == 'recipe':
                self.tts_input.append('recipe_intent_question')

            # start complex flow
            self.start_simple_command_flow()
            self.standby_handler.short_standby()

        # second phase
        else:
            nlp_output = self.last_nlp_output

            if nlp_output == 'pos':

                self.tts_input = ['incomplete_intent_response_3']

                if self.last_detected_intent == 'temp':
                    self.tts_input.append('temp_intent_example')
                elif self.last_detected_intent == 'fan':
                    if self.get_oven_state('FanOpen'):
                        self.tts_input.append('fan_intent_example_close')
                    else:
                        self.tts_input.append('fan_intent_example_open')
                elif self.last_detected_intent == 'cooking-mode':
                    self.tts_input.append('cooking_mode_intent_example')
                elif self.last_detected_intent == 'recipe':
                    self.tts_input.append('recipe_intent_example')

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

                self.start_incomplete_intent_flow()
                self.standby_handler.short_standby()


            elif nlp_output == 'neg':
                self.tts_input = 'intent_undetected_return'

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()

                self.start_incomplete_intent_flow()
                self.standby_handler.short_standby()
                self.start_incomplete_intent_flow()


            else:
                self.tts_input = 'command_unclear'

    def incomplete_intent(self):

        # still can not detect any intent
        if self.current_action == 'command_unclear' or self.current_action == '':
            return

        if self.current_action == "tam anlaşılamayan niyet" or self.current_action == "kısmen anlaşılan niyet" or self.current_action == "command unclear special third attempt":
            self.tts_input = 'intent_undetected_finished'
            self.standby_handler.intent_detected()

        self.stop_complex_missing_entity_flow()
        self.stop_incomplete_intent_flow()
        self.stop_tutorial_flow()

    def no_intent_detected(self):

        if (len(self.last_stt_output[0][1]) < 2):
            self.tts_input = "command_unclear"
            return

        # third phase (last attempt for the user)
        if self.incomplete_intent_flow:
            self.current_action = "command unclear special third attempt"
            self.is_action_inferred = True
            self.incomplete_intent()
            return

        # first phase
        if not self.tutorial_flow:
            self.tts_input = 'no_intent_detected_message_repeat'
            self.start_tutorial_flow()
            self.standby_handler.short_standby()

        else:
            if not self.simple_command_flow:
                self.tts_input = 'no_intent_detected_message_tutorial'
                self.start_simple_command_flow()
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

            else:
                if self.last_nlp_output == 'pos':
                    self.tts_input = 'tutorial_message'

                    self.stop_simple_command_flow()
                    self.standby_handler.intent_detected()

                    self.standby_handler.short_standby()
                    self.start_incomplete_intent_flow()

                elif self.last_nlp_output == 'neg':
                    self.tts_input = 'closing_message'
                    self.stop_tutorial_flow()
                    self.stop_simple_command_flow()
                    self.standby_handler.intent_detected()

                else:
                    self.tts_input = 'command_unclear'

    def interrupted_flow(self):

        if self.simple_command_flow:
            if self.last_nlp_output == 'pos':

                self.stop_simple_command_flow()
                self.complex_value_flow = self.previous_flow._flow_control["complex_value_flow"]
                self.simple_command_flow = self.previous_flow._flow_control["simple_command_flow"]

                self.complex_clock_flow = self.previous_flow._flow_control["complex_clock_flow"]
                self.recipe_loop_flow = self.previous_flow._flow_control["recipe_loop"]
                self.recipe_ingredient_flow = self.previous_flow._flow_control["recipe_ingredient"]
                self.complex_close_cooking_before_recipe_flow = self.previous_flow._flow_control[
                    'complex_close_cooking_before_recipe_flow']

                self.action_number = self.previous_flow._action_number
                self.current_action = self.previous_flow._action
                self.recipe_engine = self.previous_flow._recipe

                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

                if type(self.previous_flow._tts_input) == list:
                    if ('interrupted_flow_response_1' in self.previous_flow._tts_input):
                        self.tts_input = self.previous_flow._tts_input
                    else:
                        self.tts_input = ['interrupted_flow_response_1'] + self.previous_flow._tts_input
                else:
                    self.tts_input = ['interrupted_flow_response_1', self.previous_flow._tts_input]
                self.is_flow_interrupted = False


            elif self.last_nlp_output == 'neg':
                self.tts_input = 'interrupted_flow_response_2'
                self.is_flow_interrupted = False
                self.stop_simple_command_flow()
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()

            else:
                self.tts_input = 'command_unclear'

        else:
            self.printlog('\n*******************')
            self.printlog(f"Detected incomplete dialogue in : {self.previous_flow._action} (action)")
            self.printlog('*******************\n')
            self.tts_input = 'interrupted_flow_response'
            self.start_simple_command_flow()
            self.standby_handler.short_standby()

    def last_recipe_feedback_flow(self):

        # third stage
        if self.rating_flow:
            nlp_output = self.last_nlp_output
            if nlp_output == 'cancel' or nlp_output == "exit":
                self.tts_input = 'closing_message'
                self.stop_rating_flow()
                self.standby_handler.intent_detected()
            elif nlp_output == "invalid-value":
                self.standby_handler.intent_detected()

                if self.flow_counter > 0:
                    self.stop_rating_flow()
                    self.tts_input = "feedback_unclear_closing"

                else:
                    self.tts_input = "feedback_unclear"
                    self.flow_counter += 1
                    self.standby_handler.short_standby()



            # proper numerical value between 1 and 5
            elif type(nlp_output) == tuple:
                self.tts_input = "feedback_successful"
                self.standby_handler.intent_detected()
                self.stop_rating_flow()
                self.recipe_engine.save_rating_for_last_recipe(nlp_output[1])
            else:
                self.tts_input = "command_unclear"

        elif not self.simple_command_flow:
            self.recipe_engine.load_last_active_recipe()

            self.tts_input = ["last_recipe_feedback_question_1", self.recipe_engine.current_recipe.title,
                              "last_recipe_feedback_question_2"]
            self.start_simple_command_flow()
            self.standby_handler.short_standby()
        else:
            if self.last_nlp_output == 'pos':
                self.tts_input = "recipe_evaluation_question"
                self.stop_simple_command_flow()
                self.standby_handler.intent_detected()

                self.standby_handler.short_standby()
                self.start_rating_flow()

            elif self.last_nlp_output == 'neg':
                self.tts_input = 'closing_message'
                self.stop_simple_command_flow()
                self.standby_handler.intent_detected()

            else:
                self.tts_input = "command_unclear"

    def provide_cooking_information(self):

        now = time.time()
        cooking_time = now - self.get_oven_state('CookingStartingTime')
        cooking_duration_seconds = self.get_oven_state("CookingDuration") * 60
        remaining_cooking_seconds = cooking_duration_seconds - cooking_time
        mins, secs = divmod(remaining_cooking_seconds, 60)
        mins, secs = int(mins), int(secs)

        self.tts_input = ["remaining_cooking_info_1", f"{mins}", "remaining_cooking_info_2"]
        self.standby_handler.short_standby()

    def provide_preheating_information(self):
        self.tts_input = "preheating_info"
        self.standby_handler.short_standby()

    def register_speaker(self):
        if self.user_database.is_user_registered():
            self.tts_input = "user_already_registered"
            self.start_simple_command_flow()
            self.standby_handler.short_standby()
            self.current_action = "register_speaker_go_to_health_info"

        else:
            self.tts_input = 'register_message'
            self.current_action = "register_user_start_confirmation"
            self.start_simple_command_flow()

    def register_speaker_go_to_health_info(self):
        if self.last_nlp_output == "pos":
            self.standby_handler.intent_detected()
            self.stop_simple_command_flow()
            self.current_action = "health_information_modification"
            self.health_information_modification()
        elif self.last_nlp_output == "neg":
            self.tts_input = "closing_message"
            self.standby_handler.intent_detected()
            self.stop_simple_command_flow()

        else:
            self.tts_input = "command_unclear"

    def health_information_inquiry(self):
        if self.user_database.is_user_registered():
            self.current_action = "health_information_modification"
            self.health_information_modification()

        else:
            self.tts_input = ['user_not_registered', 'register_message']
            self.current_action = "register_user_start_confirmation"
            self.start_simple_command_flow()

    def register_user_start_confirmation(self):
        if self.last_nlp_output == "start":
            self.standby_handler.intent_detected()
            self.current_action = "user_registeration_process"
            self.tts_input = "speaker_recognition_sentence"
            self.stop_simple_command_flow()
            self.start_user_registeration_flow()
        elif self.last_nlp_output == "cancel":
            self.standby_handler.intent_detected()
            self.tts_input = "closing_message"
            self.stop_simple_command_flow()

        else:
            self.tts_input = "command_unclear"

    def user_registeration_process(self):

        if self.last_nlp_output == "sr_sentence":
            self.standby_handler.intent_detected()
            self.standby_handler.short_standby()

            # returns true if registration is complete
            if self.user_database.new_register_process():

                self.start_ingredient_flow()
                self.current_action = "add_user_ingredient"
                self.stop_user_registeration_flow()

                self.tts_input = ["user_registered_successfully", "add_user_ingredient_message"]
            else:
                self.tts_input = "speaker_recognition_sentence"


        elif self.last_nlp_output == "cancel" or self.last_nlp_output == "exit":
            self.standby_handler.intent_detected()
            self.stop_user_registeration_flow()
            self.tts_input = "closing_message"
        else:
            self.tts_input = "command_unclear"

    def health_information_modification(self):
        if not self.simple_command_flow:
            self.start_simple_command_flow()
            self.standby_handler.short_standby()

            ingredients = self.user_database.get_active_user_ingredients()

            if len(ingredients) > 0:
                self.tts_input = ["user_ingredients_list_1"]
                for ingredient in ingredients:
                    self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']

                self.tts_input.pop(-1)
                self.tts_input += ["user_ingredients_list_2"]
            else:
                self.tts_input = "no_info_registered_for_user"


        else:
            nlp_output = self.last_nlp_output

            if nlp_output == "add":
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()
                self.stop_simple_command_flow()
                self.start_ingredient_flow()
                self.current_action = "add_user_ingredient"
                self.tts_input = "add_user_ingredient_message"

            elif nlp_output == "remove":
                self.standby_handler.intent_detected()
                self.standby_handler.short_standby()
                self.stop_simple_command_flow()
                self.start_ingredient_flow()
                self.current_action = "remove_user_ingredient"
                self.tts_input = "remove_user_ingredient_message"

            elif nlp_output == "cancel" or nlp_output == "exit":
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()
                self.tts_input = 'closing_message'
            else:
                self.tts_input = "command_unclear"

    def add_user_ingredient(self):
        if not self.simple_command_flow:
            if self.last_nlp_output != -1 and self.last_nlp_output != "":
                self.stop_ingredient_flow()
                self.start_simple_command_flow()
                self.user_database.set_active_ingredient(self.last_nlp_output)

                self.tts_input = ["add_user_ingredient_confirmation_1"]

                for ingredient in self.last_nlp_output:
                    self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']

                self.tts_input.pop(-1)
                self.tts_input += ["add_user_ingredient_confirmation_2"]

            else:
                self.tts_input = 'command_unclear'

        else:

            if self.last_nlp_output == "pos":
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()
                self.tts_input = "user_ingredients_changed"
                self.user_database.add_active_ingredients()


            elif self.last_nlp_output == "neg":

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()
                self.tts_input = "user_ingredients_unchanged_exiting"

            else:
                self.tts_input = "command_unclear"

    def remove_user_ingredient(self):
        if not self.simple_command_flow:
            if self.last_nlp_output != -1 and self.last_nlp_output != "":
                self.stop_ingredient_flow()
                self.start_simple_command_flow()
                self.user_database.set_active_ingredient(self.last_nlp_output)

                self.tts_input = ["remove_user_ingredient_confirmation_1"]

                for ingredient in self.last_nlp_output:
                    self.tts_input += [f'ingredient_vocab/{ingredient}', 'and_keyword']

                self.tts_input.pop(-1)
                self.tts_input += ["remove_user_ingredient_confirmation_2"]

            else:
                self.tts_input = "command_unclear"

        else:

            if self.last_nlp_output == "pos":
                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()
                self.tts_input = "user_ingredients_changed"
                self.user_database.remove_active_ingredients()


            elif self.last_nlp_output == "neg":

                self.standby_handler.intent_detected()
                self.stop_simple_command_flow()
                self.tts_input = "user_ingredients_unchanged_exiting"

            else:
                self.tts_input = "command_unclear"

    def unregister_speaker(self):
        has_favorite_recipes = self.recipe_engine.load_favorite_recipes()
        user_is_registered = self.user_database.is_user_registered()

        # nothing to delete
        if not has_favorite_recipes and not user_is_registered:
            self.tts_input = "no_registered_information_found"

        else:

            self.tts_input = []

            if user_is_registered:
                self.tts_input += ["will_delete_audio_profile"]

            if has_favorite_recipes:
                self.tts_input += ["will_delete_favorite_recipes"]

            self.tts_input += ["confirm_before_delete_registered_information"]

            self.current_action = "confirm_unregistration"
            self.start_simple_command_flow()

    def confirm_unregistration(self):
        if self.last_nlp_output == "pos":

            self.user_database.remove_if_user_active()
            self.recipe_engine.delete_all_favorites()

            self.tts_input = "registered_info_deleted"
            self.standby_handler.intent_detected()
            self.stop_simple_command_flow()

        elif self.last_nlp_output == "neg":
            self.tts_input = "registered_info_unchanged"
            self.standby_handler.intent_detected()
            self.stop_simple_command_flow()

        else:
            self.tts_input = "command_unclear"
