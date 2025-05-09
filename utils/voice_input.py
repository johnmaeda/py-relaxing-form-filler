import os
import time
import requests
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import curses
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Attempt to load environment variables from .env file
# This should be one of the first things to do
try:
    from dotenv import load_dotenv
    load_dotenv() # Loads variables from .env into environment
    # print("DEBUG: .env file loaded successfully by dotenv in voice_input.") # Optional: for debugging
except ImportError:
    # dotenv is not installed, proceed without it (rely on system env vars)
    # print("DEBUG: dotenv not found in voice_input, relying on system environment variables.") # Optional: for debugging
    pass

# Initialize console for rich text output
console = Console()

def record_until_silence(filename="input.wav", max_duration=120, silence_duration=1.5, fs=16000, threshold=50):
    """
    Record audio until silence is detected or max duration reached. Press spacebar to stop manually.
    
    Args:
        filename (str): Path to save the recorded audio
        max_duration (float): Maximum recording duration in seconds
        silence_duration (float): Duration of silence needed to auto-stop in seconds
        fs (int): Sample rate in Hz
        threshold (float): Threshold for silence detection
        
    Returns:
        str: Path to the saved audio file
    """
    
    # Initialize curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)  # Enable special keys
    stdscr.nodelay(True)  # Make getch non-blocking
    
    try:
        # Setup
        height, width = stdscr.getmaxyx()
        meter_width = min(50, width - 10)
        max_frames = int(fs * max_duration)
        audio_buffer = np.zeros((max_frames, 1), dtype=np.int16)
        frame_count = 0
        
        # Silence detection variables
        is_silent = False
        silence_frames = 0
        silence_threshold_frames = int(silence_duration * fs)
        should_stop = False
        max_rms_seen = 1.0
        
        # Room tone calibration and display stabilization
        room_tone_baseline = None
        calibration_samples = []
        is_calibrating = True
        last_display_update = time.time()
        display_update_interval = 0.1  # Update display every 100ms
        last_meter_level = 0
        smoothed_volume = 0
        volume_history = []
        
        def callback(indata, frames, time_info, status):
            nonlocal frame_count, is_silent, silence_frames, should_stop, max_rms_seen
            nonlocal room_tone_baseline, calibration_samples, is_calibrating, last_display_update
            nonlocal last_meter_level, smoothed_volume, volume_history
            
            # Always store audio data regardless of display updates
            if frame_count + frames <= max_frames:
                audio_buffer[frame_count:frame_count+frames] = indata
                frame_count += frames
            
            try:
                # Calculate volume (RMS)
                rms = np.sqrt(np.mean(indata**2))
                
                # Safety check for NaN or infinity
                if np.isnan(rms) or np.isinf(rms):
                    rms = 0.0
                
                # Room tone calibration (first second)
                if is_calibrating:
                    calibration_samples.append(rms)
                    
                    # Show calibration message
                    if time.time() - last_display_update > display_update_interval:
                        stdscr.clear()
                        stdscr.border()
                        stdscr.addstr(1, 2, "Calibrating microphone... please stay silent")
                        stdscr.addstr(2, 2, f"Collecting room tone samples: {len(calibration_samples)}")
                        stdscr.refresh()
                        last_display_update = time.time()
                    
                    # After 1 second, calculate baseline
                    if len(calibration_samples) >= fs / frames:
                        is_calibrating = False
                        # Calculate room tone (with a small margin)
                        room_tone_baseline = np.mean(calibration_samples) * 1.2
                        # Adjust threshold based on room tone if needed
                        adjusted_threshold = max(threshold, room_tone_baseline * 1.5)
                        # Reset the display
                        stdscr.clear()
                        stdscr.border()
                        stdscr.addstr(1, 2, "Room tone calibration complete!")
                        stdscr.addstr(2, 2, f"Baseline noise: {room_tone_baseline:.1f}")
                        stdscr.addstr(3, 2, "Recording started...")
                        stdscr.refresh()
                        time.sleep(0.5)  # Brief pause to show calibration completed
                    
                    return  # Skip the rest during calibration
                
                # Update volume history with a moving window (last 5 values)
                volume_history.append(rms)
                if len(volume_history) > 5:
                    volume_history.pop(0)
                
                # Smooth the volume using exponential moving average
                alpha = 0.3  # Smoothing factor (0.3 = 30% new value, 70% old value)
                smoothed_volume = alpha * rms + (1 - alpha) * smoothed_volume
                
                # Update max RMS seen (with safety check)
                if smoothed_volume > 200 and not np.isnan(smoothed_volume):
                    max_rms_seen = max(max_rms_seen, smoothed_volume)
                
                # Make sure max_rms_seen is never too small
                if max_rms_seen < 0.1:
                    max_rms_seen = 1.0
                
                # Check for silence based on smoothed volume
                if smoothed_volume < threshold:
                    silence_frames += frames
                    if not is_silent:
                        is_silent = True
                else:
                    silence_frames = 0
                    if is_silent:
                        is_silent = False
                
                # Calculate meter level with smoothing
                new_meter_level = int((smoothed_volume / max_rms_seen) * meter_width)
                new_meter_level = max(0, min(new_meter_level, meter_width))
                
                # Only update display at intervals or if significant change in meter level
                current_time = time.time()
                if (current_time - last_display_update > display_update_interval or 
                    abs(new_meter_level - last_meter_level) > 3):
                    
                    # Clear screen and draw borders
                    stdscr.clear()
                    stdscr.border()
                    
                    # Display recording info
                    stdscr.addstr(1, 2, f"Recording... (max {max_duration}s, stops after {silence_duration}s silence)")
                    stdscr.addstr(2, 2, f"Elapsed: {frame_count/fs:.1f}s")
                    
                    # Show instructions for manual stop
                    stdscr.addstr(3, 2, "Press SPACEBAR to stop recording manually")
                    
                    # Show silence status
                    if is_silent:
                        silence_seconds = silence_frames / fs
                        stdscr.addstr(4, 2, f"Silence detected: {silence_seconds:.1f}s / {silence_duration}s")
                    else:
                        stdscr.addstr(4, 2, "Speaking detected")
                    
                    # Draw audio meter
                    meter_y = 6
                    stdscr.addstr(meter_y, 2, "[" + "=" * new_meter_level + " " * (meter_width - new_meter_level) + "]")
                    
                    # Show RMS value
                    stdscr.addstr(meter_y + 1, 2, f"Volume: {smoothed_volume:.1f}" + 
                                 (f" (Room tone: {room_tone_baseline:.1f})" if room_tone_baseline else ""))
                    
                    # Show threshold line
                    threshold_position = int((threshold / max_rms_seen) * meter_width)
                    threshold_position = min(max(threshold_position, 0), meter_width)
                    if 2 + threshold_position < width - 1:
                        stdscr.addstr(meter_y, 2 + threshold_position, "|")
                        stdscr.addstr(meter_y + 2, 2, f"Threshold: {threshold}")
                    
                    # Refresh the display
                    stdscr.refresh()
                    last_display_update = current_time
                    last_meter_level = new_meter_level
                
                # Stop if silence threshold reached
                if silence_frames >= silence_threshold_frames:
                    stdscr.clear()
                    stdscr.border()
                    stdscr.addstr(1, 2, "Silence threshold reached, stopping...")
                    stdscr.refresh()
                    should_stop = True
                    raise sd.CallbackStop
                
                # Stop if max duration reached
                if frame_count >= max_frames:
                    stdscr.clear() 
                    stdscr.border()
                    stdscr.addstr(1, 2, "Max duration reached, stopping...")
                    stdscr.refresh()
                    should_stop = True
                    raise sd.CallbackStop
                
            except Exception as e:
                # Handle any errors in the callback safely
                stdscr.clear()
                stdscr.addstr(1, 2, f"Error in callback: {str(e)}")
                stdscr.refresh()

        # Start recording
        stream = sd.InputStream(samplerate=fs, channels=1, dtype=np.int16, callback=callback)
        with stream:
            stdscr.addstr(1, 2, "Starting recording...")
            stdscr.refresh()
            
            # Main loop - check for early termination or spacebar press
            end_time = time.time() + max_duration
            while time.time() < end_time and not should_stop:
                # Check for spacebar press (32 is the ASCII code for space)
                try:
                    key = stdscr.getch()
                    if key == 32:  # Spacebar
                        stdscr.clear()
                        stdscr.border()
                        stdscr.addstr(1, 2, "Manually stopped recording...")
                        stdscr.refresh()
                        should_stop = True
                        break
                except:
                    pass
                
                sd.sleep(50)  # Check conditions more frequently
        
        # Show final message
        stdscr.clear()
        stdscr.addstr(1, 2, "Recording complete!")
        stdscr.addstr(2, 2, f"Duration: {frame_count/fs:.1f} seconds")
        stdscr.addstr(3, 2, f"Saving to {filename}...")
        stdscr.refresh()
        time.sleep(1)
        
        # Trim and save the audio
        final_audio = audio_buffer[:frame_count]
        write(filename, fs, final_audio)
        
        return filename
    
    finally:
        # Clean up curses
        curses.echo()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.endwin()

def transcribe_audio_mlx(filename, model_name="mlx-community/whisper-base.en-mlx"):
    """
    Transcribe audio using MLX Whisper locally.
    
    Args:
        filename (str): Path to the audio file to transcribe
        model_name (str): Name of the MLX Whisper model to use
        
    Returns:
        str: Transcribed text
    """
    if not MLX_WHISPER_AVAILABLE:
        raise ImportError("mlx-whisper package is not installed. Install it with 'pip install mlx-whisper'")
    
    console.print("\n[dim]Transcribing audio with MLX Whisper...[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold green]Processing with {model_name}...[/bold green]"),
        transient=True,
    ) as progress:
        task = progress.add_task("Transcribing", total=None)
        result = mlx_whisper.transcribe(filename, path_or_hf_repo=model_name)
        progress.update(task, completed=True)
    
    return result["text"]

def transcribe_audio(filename, azure_endpoint=None, azure_api_key=None, deployment_name="whisper", 
                    api_version="2024-02-01", mode="azure", mlx_model="mlx-community/whisper-base.en-mlx"):
    """
    Transcribe audio using Azure OpenAI or local MLX Whisper.
    
    Args:
        filename (str): Path to the audio file to transcribe
        azure_endpoint (str, optional): Azure OpenAI endpoint. If None, uses AZURE_OPENAI_ENDPOINT env var
        azure_api_key (str, optional): Azure OpenAI API key. If None, uses AZURE_OPENAI_API_KEY env var
        deployment_name (str, optional): Deployment name for the Whisper model. Defaults to "whisper"
        api_version (str, optional): API version to use. Defaults to "2024-02-01"
        mode (str): Transcription mode - "azure" or "local" (using MLX)
        mlx_model (str): MLX Whisper model to use when mode is "local"
        
    Returns:
        str: Transcribed text
    """
    if mode == "local":
        return transcribe_audio_mlx(filename, model_name=mlx_model)
    
    # Azure mode
    # Get credentials from environment if not provided
    endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
    
    # Check for required credentials
    if not endpoint or not api_key:
        raise ValueError("Azure OpenAI endpoint and API key are required. "
                        "Please provide them as arguments or set AZURE_OPENAI_ENDPOINT "
                        "and AZURE_OPENAI_API_KEY environment variables.")
    
    # Remove trailing slash if present
    endpoint = endpoint.rstrip("/")
    
    url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version={api_version}"
    headers = {"api-key": api_key}

    with open(filename, "rb") as f:
        files = {"file": (filename, f, "audio/wav")}
        
        console.print("\n[dim]Transcribing audio with Azure OpenAI...[/dim]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Processing audio...[/bold green]"),
            transient=True,
        ) as progress:
            task = progress.add_task("Transcribing", total=None)
            response = requests.post(url, headers=headers, files=files)
            progress.update(task, completed=True)
        
        response.raise_for_status()
        transcription = response.json()["text"]
        return transcription

def record_and_transcribe(output_file="input.wav", max_duration=120, silence_duration=1.5, 
                         azure_endpoint=None, azure_api_key=None, mode="azure", 
                         mlx_model="mlx-community/whisper-base.en-mlx"):
    """
    Record audio from the microphone and transcribe it using Azure OpenAI or local MLX Whisper.
    
    Args:
        output_file (str): Path to save the recorded audio
        max_duration (float): Maximum recording duration in seconds
        silence_duration (float): Duration of silence needed to auto-stop in seconds
        azure_endpoint (str, optional): Azure OpenAI endpoint. If None, uses AZURE_OPENAI_ENDPOINT env var
        azure_api_key (str, optional): Azure OpenAI API key. If None, uses AZURE_OPENAI_API_KEY env var
        mode (str): Transcription mode - "azure" or "local" (using MLX)
        mlx_model (str): MLX Whisper model to use when mode is "local"
        
    Returns:
        str: Transcribed text
    """
    try:
        # Record audio
        console.print("🔴 [italic blue]...recording audio...[/italic blue]")
        audio_file = record_until_silence(
            filename=output_file,
            max_duration=max_duration,
            silence_duration=silence_duration
        )
        
        # Transcribe audio
        transcription = transcribe_audio(
            filename=audio_file,
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            mode=mode,
            mlx_model=mlx_model
        )
        
        # console.print(f"\n[bold green]Transcription:[/bold green] {transcription}") # Commented out to prevent double printing
        return transcription
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Error:[/bold red] {str(e)}")
        return None

# Example usage if this file is run directly
if __name__ == "__main__":
    try:
        # Check for required environment variables or MLX availability
        mode = "local" if MLX_WHISPER_AVAILABLE else "azure"
        
        if mode == "azure" and (not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT")):
            console.print("[bold red]Error: Azure mode requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.[/bold red]")
            
            if MLX_WHISPER_AVAILABLE:
                console.print("[bold yellow]Falling back to local MLX Whisper mode.[/bold yellow]")
                mode = "local"
            else:
                console.print("[bold red]MLX Whisper is not available. Install with 'pip install mlx-whisper'[/bold red]")
                exit(1)
            
        # Record and transcribe
        transcription = record_and_transcribe(mode=mode)
        
        if transcription:
            console.print("\n[bold blue]Transcription successful![/bold blue]")
        else:
            console.print("\n[bold red]Transcription failed.[/bold red]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error: {str(e)}[/bold red]") 