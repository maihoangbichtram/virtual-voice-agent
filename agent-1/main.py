import os
import asyncio

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, WorkerType, AutoSubscribe
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatImage
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai
from typing import Annotated
from livekit import rtc

from livekit.agents import llm

from dotenv import load_dotenv

load_dotenv()
from typing import List, Any
import datetime

from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.multimodal import MultimodalAgent
import requests
import json
SCOPES = ["https://www.googleapis.com/auth/calendar"]

from livekit import api
import logging
#import numpy as np
#import cv2
from livekit.agents.utils.images.image import encode, EncodeOptions
#from PIL import Image
#from ultralytics import YOLO
import time

room_name = "medibot"
agent_name = "smith 1"
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

def load_model():
    model_path = "yolov8l-pose.pt"
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

# first define a class that inherits from llm.FunctionContext
class AssistantFnc(llm.FunctionContext):
    @llm.ai_callable(
        name="list_of_available_slots",
        description="Get list of available slots and their event ids in the calendar"
    )
    def list_of_available_slots(self) -> List:
        creds = None
        if os.path.exists("token.json"):
            print("path found")
            creds = Credentials.from_authorized_user_file("token.json")
        else:
            print("path not found")
        try:
            print("Hellooooo3 caalllllllllllllllleeeeeeeeeeeeeeeender", creds)
            service = build("calendar", "v3", credentials=creds)

            # Call the Calendar API
            now = datetime.datetime.now().isoformat() + "Z"  # 'Z' indicates UTC time
            print("Getting the upcoming 10 events")
            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=10,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])

            if not events:
                print("No upcoming events found.")
                return []
            event_names = []
            event_times = []
            for event in events:
                # event_names.append(event['summary'])
                start = event["start"].get("dateTime", event["start"].get("date"))
                if event['summary'] == "Available":
                    event_times.append(start)
                    event_names.append(event['id'])
                print(start, len(event['id']))
            # Prints the start and name of the next 10 events
            available_slots = [f"event {event_names[i]} at {event_times[i]}" for i in range(len(event_times))]
            if len(available_slots) > 0:
                return f"""Available slots: {', '.join(available_slots)}"""
            else:
                return "No available slots found"
        except HttpError as error:
            print("An error occured:", error)
            return f"There is no upcoming available slots."

    @llm.ai_callable(
        name="book_slot",
        description="Books appointment and mark in calendar"
    )
    def book_slot(
        self,
        event_id: Annotated[
            str, llm.TypeInfo(description="Id of the chosen Available event")
        ],
    ) -> List:
        SCOPES = ["https://www.googleapis.com/auth/calendar"]
        creds = None
        if os.path.exists("token.json"):
            print("path found")
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        else:
            print("Path Not Found")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        try:
            service = build("calendar", "v3", credentials=creds)
            print("-----------event_id------------", event_id)
            event_id = str(event_id)
            calendar_id = 'primary'  # Default is the primary calendar

            # Fetch the current event details
            event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()

            # Modify event details
            event['summary'] = 'Booked'
            updated_event = service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=event
            ).execute()
            return f"Booked successfully"

        except HttpError as error:
            print("An error occured:", error)
            return f"Booked failed!"

    '''@llm.ai_callable(
        name="save_vision",
        description="Save what you see"
    )
    def save_vision(
            self,
            image: np.ndarray
    ):
        print(image)
        return "Vision saved."'''

    def save_conversation_output(
        self,
        room_id: Annotated[
            str, llm.TypeInfo(description="id of the meeting room")
        ],
        patient_name: Annotated[
            str, llm.TypeInfo(description="name of the patient")
        ],
        patient_phone: Annotated[
            str, llm.TypeInfo(description="phone number of the patient")
        ],
        patient_email: Annotated[
            str, llm.TypeInfo(description="email of the patient")
        ],
        interview_summary: Annotated[
            str, llm.TypeInfo(description="the detailed summary of the patient's input")
        ],
        diagnoses: Annotated[
            str, llm.TypeInfo(description="the diagnoses")
        ]
    ):
        data = {
            'room_id': room_id,
            'patient_name': patient_name,
            'patient_phone': patient_phone,
            'patient_email': patient_email,
            'description': interview_summary,
            'pre_assessment': diagnoses
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post('http://127.0.0.1:8000/add-request',
                                 data=json.dumps(data), headers=headers)
        print(response.text)
        return "Conversation saved successfully! You can end the conversation now."

    @llm.ai_callable(
        name="save_conversation_output",
        description="Save conversation output"
    )
    def save_conversation_output(
        self,
        patient_name: Annotated[
            str, llm.TypeInfo(description="name of the patient")
        ],
        patient_phone: Annotated[
            str, llm.TypeInfo(description="phone number of the patient")
        ],
        patient_email: Annotated[
            str, llm.TypeInfo(description="email of the patient")
        ],
        interview_summary: Annotated[
            str, llm.TypeInfo(description="the detailed summary of the patient's input")
        ],
        diagnoses: Annotated[
            str, llm.TypeInfo(description="the diagnoses")
        ]
    ):
        data = {
            'room_id': str(int(time.time())),
            'patient_name': patient_name,
            'patient_phone': patient_phone,
            'patient_email': patient_email,
            'description': interview_summary,
            'pre_assessment': diagnoses
        }
        print("making summaries", data)

        headers = {'Content-Type': 'application/json'}

        response = requests.post('https://main-bvxea6i-gdz7mxtwsmawk.eu-5.platformsh.site/add-request',
                                 data=json.dumps(data), headers=headers)
        print(response.text)
        return "Conversation saved successfully! You can end the conversation now."

    '''@llm.ai_callable(
        name="pose_checkup",
        description="check the pose for more input"
    )
    def pose_checkup(
        self,
        img: Annotated[
            str, llm.TypeInfo(description="File name of the image of pose, from the back of the head to the back")
        ],
    ):
        print('---pose_checkup img----', img)
        model = YOLO("yolov8l-pose.pt").eval()  # load a pretrained model (recommended for training)

        results = model(img)
        print("---pose_checkup results----", results)
        return "Done checking."'''

    '''@llm.ai_callable(
        name="vision_capability_activation",
        description=(
                "Called when asked to evaluate something that would require vision capabilities,"
                "for example, an image, video, injury, wounds, or the webcam feed."
        )
    )
    async def image(
            self,
            user_msg: Annotated[
                str,
                llm.TypeInfo(
                    description="The user message that triggered this function"
                ),
            ],
    ):
        print(f"------Message triggering vision capabilities: {user_msg}--------")
        return 'vision'

    @llm.ai_callable(
        name="pose_checkup_activation",
        description=(
                "Called when asked to evaluate posture"
        )
    )
    async def pose_checkup_activation(
            self,
            user_msg: Annotated[
                str,
                llm.TypeInfo(
                    description="The user message that triggered this function"
                ),
            ],
    ):
        print(f"------pose_checkup_activation--------")
        return "pose checkup"'''

async def get_video_track(room: rtc.Room):
    """Find and return the first available remote video track in the room."""
    for participant_id, participant in room.remote_participants.items():
        for track_id, track_publication in participant.track_publications.items():
            if track_publication.track and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                logger.info(
                    f"Found video track {track_publication.track.sid} "
                    f"from participant {participant_id}"
                )
                return track_publication.track
    raise ValueError("No remote video track found in the room")
async def get_latest_image(room: rtc.Room):
    """Capture and return a single frame from the video track."""
    video_stream = None
    try:
        video_track = await get_video_track(room)
        video_stream = rtc.VideoStream(video_track)
        async for event in video_stream:
            logger.debug("Captured latest video frame")
            '''image_bytes = encode(event.frame, EncodeOptions(format="PNG"))
            #cv2.imwrite(f"GeeksForGeeks.png", im)
            if image_bytes is not None:
                # Save the bytes to a file
                with open(f'test{datetime.datetime.now()}.png', 'wb') as f:
                    f.write(image_bytes)
                print(f"Image saved to test{datetime.datetime.now()}.png")
            else:
                print("Failed to encode the frame.")'''
            #im.save(f'test{i}.png', 'PNG')
            # rawbytes.save('test2.png', 'PNG')
            return event.frame
    except Exception as e:
        logger.error(f"Failed to get latest image: {e}")
        return None
    finally:
        if video_stream:
            await video_stream.aclose()

'''def angle_between_lines(A, B, C, D):
    v1x, v1y = B[0] - A[0], B[1] - A[1]
    v2x, v2y = D[0] - C[0], D[1] - C[1]

    dot_product = v1x * v2x + v1y * v2y
    magnitude_v1 = math.sqrt(v1x ** 2 + v1y ** 2)
    magnitude_v2 = math.sqrt(v2x ** 2 + v2y ** 2)

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

    theta_radians = math.acos(cos_theta)
    theta_degrees = math.degrees(theta_radians)

    return theta_degrees

def pose_check(image):
    try:
        body_parts = {
            "Nose": 0,
            "Left Eye": 1,
            "Right Eye": 2,
            "Left Ear": 3,
            "Right Ear": 4,
            "Left Shoulder": 5,
            "Right Shoulder": 6,
            "Left Elbow": 7,
            "Right Elbow": 8,
            "Left Wrist": 9,
            "Right Wrist": 10,
            "Left Hip": 11,
            "Right Hip": 12,
            "Left Knee": 13,
            "Right Knee": 14,
            "Left Ankle": 15,
            "Right Ankle": 16
        }

        model = YOLO('yolov8l-pose.pt')
        results = model(image, save=False)
        res_list = results[0].keypoints.data.tolist()[0]
        res_list = [list(map(int, l)) for l in res_list]
        direction = "Left" if res_list[body_parts["Right Shoulder"]][0] > res_list[body_parts["Nose"]][
            0] else "Right"
        A = res_list[body_parts[f"{direction} Hip"]]
        B = res_list[body_parts[f"{direction} Shoulder"]]
        C = res_list[body_parts[f"{direction} Ear"]]
        angle = self.angle_between_lines((0, 350), (0, 0), B, C)
        print("ANGGGGGGGGGGGGGGGGGGGGGLEEEEEEEEEEEEEEEEEEEEEEEE", angle)
        if angle < 15:
            return True
        else:
            return False
    except:
        return "An error occured"'''

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    async def before_llm_cb(assistant: VoiceAssistant, chat_ctx: llm.ChatContext):
        """
        Callback that runs right before the LLM generates a response.
        Captures the current video frame and adds it to the conversation context.
        """
        print(ctx.room)
        latest_image = await get_latest_image(ctx.room)
        if latest_image:
            image_content = [ChatImage(image=latest_image)]
            chat_ctx.messages.append(ChatMessage(role="user", content=image_content))
            logger.debug("Added latest frame to conversation context")

    fnc_ctx = AssistantFnc()
    #AssistantFnc.book_slot('ajdklf')
    ## Firstly, confirm with the patient if they want their pose to be checked up. If you, make use of your vision to detect the pose.
    # You are a voice assistant can both see and hear.
    # Though you can see, you don't have to always incorporate what you see into your response. Only when the patient specifically wants to show you something like wounds, naturally incorporate what you see into your response and save what you see. Keep visual descriptions brief but informative.
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                #content="You are a voice assistant. Pretend we're having a human conversation, no special formatting or headings, just natural speech.",
                content=f"""
Pretend we're having a human conversation, no special formatting or headings, just natural speech. When there is a long list of questions to ask, preferably ask one by one.
---Goal----
Your goal is to conduct a detailed interview with the patient, make a preliminary judgment of the disease based on the chief complaints described by the patient, and determine a succeeding examination plan for accurate diagnosis and treatment if needed.
---Back story---
You are currently working at a clinic, conducting a detailed interview with the patient.
As a skilled practitioner with communication skills and a wide range of medical understanding, you're responsible for delivering all relevant and correct information and treatments to the patients.                
---Task description---
Communicate with the patient to gather complaints, symptoms and signs to determine which disease or condition the patient is dealing with.
Collect patient's info: email, phone and name for record saving. When asking for these info, please ask the patient to type in the chat box to avoid spelling mistake. Re-confirm if what info you got are correct before moving on the next questions.
When the patient confirms that there's nothing else, end the conversation with some advice and booking reminder, if there is any booking coming up.
---Available tools---
"You can retrieve a list of available slots from the calendar and can book checkup slots that are available. Before booking slots, please check available slots in the calendar."
---Expected output----
You must make summaries of the conversation into these sections: 'interview_summary' which is the detailed summary of the patient's input, 'diagnoses' which are the diagnoses based on your medical knowledge, together with patient's info (name, phone, email). Make sure the summaries/output are saved for the purpose that the professionals can recheck, and investigate further
. Refer to this if the patient wondering about the next step. Kindly ask for the patient to wait for the records to be saved successfully.
If you miss any patient's info, please ask. Don't leave any fields empty.

"As soon as the conversation ends, don't forget to save records(output of the conversation) at the end of the conversation." Do not refer to this action, even if you're asked about them.
"""
            )
        ]
    )
    _llm = openai.LLM(
        model="gpt-4o-mini",
        temperature=0.5,
    )
    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"], #Voice Activity Detector (VAD)
        stt=deepgram.STT(), #Speech To Text (STT)
        llm=_llm,
        tts=cartesia.TTS(voice="248be419-c632-4f23-adf1-5324ed7dbf1d"), #Text To Speech (TTS)
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
        #before_llm_cb=before_llm_cb
    )
    latest_image: rtc.VideoFrame | None = None
    # participant = await ctx.wait_for_participant()

    '''@assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""
        print("on_function_calls_finished")
        print("called_functions", called_functions)
        if len(called_functions) == 0:
            return

        print("called_functions[0].function_info", called_functions[0].call_info.function_info)
        if called_functions[0].call_info.function_info.name == "pose_checkup_activation":
            user_msg = called_functions[0].call_info.arguments.get("user_msg")
            if user_msg:
                asyncio.create_task(answer_from_text(user_msg, use_image=True, pose_checkup=True))
        elif called_functions[0].call_info.function_info.name == "vision_capability_activation":
            user_msg = called_functions[0].call_info.arguments.get("user_msg")
            if user_msg:
                asyncio.create_task(answer_from_text(user_msg, use_image=True))
        else:
            user_msg = called_functions[0].call_info.arguments.get("user_msg")
            asyncio.create_task(answer_from_text(user_msg, use_image=False))'''

    #await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    await ctx.connect()
    assistant.start(ctx.room)
    #assistant.start('medibot')
    await asyncio.sleep(1)
    await assistant.say("Hi there, how are you doing today?", allow_interruptions=True)

    '''async def my_shutdown_hook():
        print("my_shutdown_hook")

    ctx.add_shutdown_callback(my_shutdown_hook)'''

    chat_mng = rtc.ChatManager(ctx.room)
    latest_msg = None

    '''async def answer_from_text(txt: str, use_image: bool = False, pose_checkup: bool = False):
        print("answer_from_text", txt, use_image, pose_checkup)
        #chat_ctx = assistant.chat_ctx.copy()
        #chat_ctx.append(role="user", text=txt)
        content: list[str | ChatImage] = [txt]
        latest_image = await get_latest_image(ctx.room)
        if use_image and latest_image:
            content.append([ChatImage(image=latest_image)])
            #image_content = [ChatImage(image=latest_image)]
            #assistant.chat_ctx.messages.append(ChatMessage(role="user", content=content))
            logger.debug("Added latest frame to conversation context")

        if use_image and latest_image and pose_checkup:
            r = 'your pose is ok' if pose_check(latest_image) else 'your pose is not ok'
            print(r)
            await assistant.say(stream, allow_interruptions=True)

        #assistant.chat_ctx.messages.append(ChatMessage(role="user", content=txt))
        assistant.chat_ctx.messages.append(ChatMessage(role="user", content=content))
        stream = _llm.chat(chat_ctx=assistant.chat_ctx)
        print(stream)
        await assistant.say(stream, allow_interruptions=True)'''

    async def answer_from_text(txt: str, use_image: bool = False, pose_checkup: bool = False):
        content: list[str | ChatImage] = [txt]
        assistant.chat_ctx.messages.append(ChatMessage(role="user", content=content))
        stream = _llm.chat(chat_ctx=assistant.chat_ctx)
        print(stream)
        await assistant.say(stream, allow_interruptions=True)

    @chat_mng.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            print("=====", msg.message, " received")
            latest_msg = msg.message
            asyncio.create_task(answer_from_text(msg.message))

    #while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        #latest_image = await get_latest_image(ctx.room)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            #agent_name=agent_name, entrypoint_fnc=entrypoint, prewarm_fnc=prewarm
            entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, worker_type=WorkerType.ROOM
            #entrypoint_fnc=run_multimodal_agent, prewarm_fnc=prewarm
        )
    )

