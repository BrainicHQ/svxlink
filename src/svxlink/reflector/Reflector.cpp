/**
@file	 Reflector.cpp
@brief   The main reflector class
@author  Tobias Blomberg / SM0SVX
@date	 2017-02-11

Additional VAD functionality added by Silviu Stroe (www.brainic.io) in 2024.

\verbatim
SvxReflector - An audio reflector for connecting SvxLink Servers
Copyright (C) 2003-2024 Tobias Blomberg / SM0SVX

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
\endverbatim
*/

/****************************************************************************
 *
 * System Includes
 *
 ****************************************************************************/

#include <cassert>
#include <json/json.h>


/****************************************************************************
 *
 * Project Includes
 *
 ****************************************************************************/

#include <AsyncConfig.h>
#include <AsyncTcpServer.h>
#include <AsyncUdpSocket.h>
#include <AsyncApplication.h>
#include <AsyncPty.h>
#include <common.h>
#include <codecvt>


/****************************************************************************
 *
 * Local Includes
 *
 ****************************************************************************/

#include "Reflector.h"
#include "ReflectorClient.h"
#include "TGHandler.h"
#include "opus_wrapper.h"
#include "VadIterator.h"


/****************************************************************************
 *
 * Namespaces to use
 *
 ****************************************************************************/

using namespace std;
using namespace Async;



/****************************************************************************
 *
 * Defines & typedefs
 *
 ****************************************************************************/



/****************************************************************************
 *
 * Local class definitions
 *
 ****************************************************************************/



/****************************************************************************
 *
 * Prototypes
 *
 ****************************************************************************/



/****************************************************************************
 *
 * Exported Global Variables
 *
 ****************************************************************************/



/****************************************************************************
 *
 * Local Global Variables
 *
 ****************************************************************************/

namespace {
  ReflectorClient::ProtoVerRangeFilter v1_client_filter(
      ProtoVer(1, 0), ProtoVer(1, 999));
  ReflectorClient::ProtoVerRangeFilter v2_client_filter(
      ProtoVer(2, 0), ProtoVer(2, 999));
};


/****************************************************************************
 *
 * Public member functions
 *
 ****************************************************************************/

Reflector::Reflector(void)
  : m_srv(0), m_udp_sock(0), m_tg_for_v1_clients(1), m_random_qsy_lo(0),
    m_random_qsy_hi(0), m_random_qsy_tg(0), m_http_server(0), m_cmd_pty(0)
{
  TGHandler::instance()->talkerUpdated.connect(
      mem_fun(*this, &Reflector::onTalkerUpdated));
  TGHandler::instance()->requestAutoQsy.connect(
      mem_fun(*this, &Reflector::onRequestAutoQsy));

} /* Reflector::Reflector */


Reflector::~Reflector(void)
{
  delete m_http_server;
  m_http_server = 0;
  delete m_udp_sock;
  m_udp_sock = 0;
  delete m_srv;
  m_srv = 0;
  delete m_cmd_pty;
  m_cmd_pty = 0;
  m_client_con_map.clear();
  ReflectorClient::cleanup();
  delete TGHandler::instance();
} /* Reflector::~Reflector */


bool Reflector::initialize(Async::Config &cfg)
{
  m_cfg = &cfg;
  TGHandler::instance()->setConfig(m_cfg);

    // Initialize the GCrypt library if not already initialized
  if (!gcry_control(GCRYCTL_INITIALIZATION_FINISHED_P))
  {
    gcry_check_version(NULL);
    gcry_error_t err;
    err = gcry_control(GCRYCTL_DISABLE_SECMEM, 0);
    if (err != GPG_ERR_NO_ERROR)
    {
      cerr << "*** ERROR: Failed to initialize the Libgcrypt library: "
           << gcry_strsource(err) << "/" << gcry_strerror(err) << endl;
      return false;
    }
      // Tell Libgcrypt that initialization has completed
    err = gcry_control(GCRYCTL_INITIALIZATION_FINISHED, 0);
    if (err != GPG_ERR_NO_ERROR)
    {
      cerr << "*** ERROR: Failed to initialize the Libgcrypt library: "
           << gcry_strsource(err) << "/" << gcry_strerror(err) << endl;
      return false;
    }
  }

  std::string listen_port("5300");
  cfg.getValue("GLOBAL", "LISTEN_PORT", listen_port);
  m_srv = new TcpServer<FramedTcpConnection>(listen_port);
  m_srv->clientConnected.connect(
      mem_fun(*this, &Reflector::clientConnected));
  m_srv->clientDisconnected.connect(
      mem_fun(*this, &Reflector::clientDisconnected));

  uint16_t udp_listen_port = 5300;
  cfg.getValue("GLOBAL", "LISTEN_PORT", udp_listen_port);
  m_udp_sock = new UdpSocket(udp_listen_port);
  if ((m_udp_sock == 0) || !m_udp_sock->initOk())
  {
    cerr << "*** ERROR: Could not initialize UDP socket" << endl;
    return false;
  }
  m_udp_sock->dataReceived.connect(
      mem_fun(*this, &Reflector::udpDatagramReceived));

  unsigned sql_timeout = 0;
  cfg.getValue("GLOBAL", "SQL_TIMEOUT", sql_timeout);
  TGHandler::instance()->setSqlTimeout(sql_timeout);

  unsigned sql_timeout_blocktime = 60;
  cfg.getValue("GLOBAL", "SQL_TIMEOUT_BLOCKTIME", sql_timeout_blocktime);
  TGHandler::instance()->setSqlTimeoutBlocktime(sql_timeout_blocktime);

  m_cfg->getValue("GLOBAL", "TG_FOR_V1_CLIENTS", m_tg_for_v1_clients);

  SvxLink::SepPair<uint32_t, uint32_t> random_qsy_range;
  if (m_cfg->getValue("GLOBAL", "RANDOM_QSY_RANGE", random_qsy_range))
  {
    m_random_qsy_lo = random_qsy_range.first;
    m_random_qsy_hi = m_random_qsy_lo + random_qsy_range.second-1;
    if ((m_random_qsy_lo < 1) || (m_random_qsy_hi < m_random_qsy_lo))
    {
      cout << "*** WARNING: Illegal RANDOM_QSY_RANGE specified. Ignored."
           << endl;
      m_random_qsy_hi = m_random_qsy_lo = 0;
    }
    m_random_qsy_tg = m_random_qsy_hi;
  }

  std::string http_srv_port;
  if (m_cfg->getValue("GLOBAL", "HTTP_SRV_PORT", http_srv_port))
  {
    m_http_server = new Async::TcpServer<Async::HttpServerConnection>(http_srv_port);
    m_http_server->clientConnected.connect(
        sigc::mem_fun(*this, &Reflector::httpClientConnected));
    m_http_server->clientDisconnected.connect(
        sigc::mem_fun(*this, &Reflector::httpClientDisconnected));
  }
    
    // Path for command PTY
    string pty_path;
    m_cfg->getValue("GLOBAL", "COMMAND_PTY", pty_path);
    if (!pty_path.empty())
    {
        m_cmd_pty = new Pty(pty_path);
        if ((m_cmd_pty == nullptr) || !m_cmd_pty->open())
        {
            std::cerr << "*** ERROR: Could not open command PTY '" << pty_path
            << "' as specified in configuration variable "
            "GLOBAL/COMMAND_PTY" << std::endl;
            return false;
        }
        m_cmd_pty->setLineBuffered(true);
        m_cmd_pty->dataReceived.connect(
                                        mem_fun(*this, &Reflector::ctrlPtyDataReceived));
    }
    
    m_cfg->valueUpdated.connect(sigc::mem_fun(*this, &Reflector::cfgUpdated));
    
    

    m_cfg->getValue("VAD_SETTINGS", "VAD_ENABLED_CALLSIGNS", vadEnabledCallsigns); // Get the list of
    // callsigns in format "callsign1,callsign2,callsign3"
    m_cfg->getValue("VAD_SETTINGS", "SILERO_MODEL_PATH", modelPath); // Get the path to the ONNX model
    // Convert std::string to std::wstring
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wideModelPath = converter.from_bytes(modelPath);
    // Now you can pass wideModelPath to a constructor or function that requires std::wstring

    std::string isVadEnabledStr; // Temporary string to hold the value
    if (m_cfg->getValue("VAD_SETTINGS", "IS_VAD_ENABLED", isVadEnabledStr)) {
        // Convert the string value to a boolean
        isVadEnabled = (isVadEnabledStr == "true");
    } else {
        // Handle the case where the value could not be retrieved, possibly set a default value
        isVadEnabled = false; // Default to false or true depending on your application's needs
    }
    m_cfg->getValue("VAD_SETTINGS", "SAMPLE_RATE", sampleRate); // Get the sample rate
    m_cfg->getValue("VAD_SETTINGS", "WINDOW_SIZE_SAMPLES", windowSizeSamples); // Get the sample rate
    m_cfg->getValue("VAD_SETTINGS", "THRESHOLD", threshold); // Get the threshold value
    m_cfg->getValue("VAD_SETTINGS", "PROCESSED_SAMPLE_BUFFER_SIZE", sampleBufferSize); // Get the sample buffer size (in samples
    m_cfg->getValue("VAD_SETTINGS", "VAD_GATE_SAMPLE_SIZE", vadGateSampleSize); // Get the sample buffer size (in samples)
    m_cfg->getValue("VAD_SETTINGS", "START_SILENCE_REPLACEMENT_BUFFER_MS", startSilenceMs); // Replace the first ms of the buffer with silence

    if(isVadEnabled) {
    vadIterator = std::make_unique<VadIterator>(wideModelPath, sampleRate,
                                                windowSizeSamples, threshold);
    }

  return true;
} /* Reflector::initialize */


void Reflector::nodeList(std::vector<std::string>& nodes) const
{
  nodes.clear();
  for (const auto& item : m_client_con_map)
  {
    const std::string& callsign = item.second->callsign();
    if (!callsign.empty())
    {
      nodes.push_back(callsign);
    }
  }
} /* Reflector::nodeList */


void Reflector::broadcastMsg(const ReflectorMsg& msg,
                             const ReflectorClient::Filter& filter)
{
  for (const auto& item : m_client_con_map)
  {
    ReflectorClient *client = item.second;
    if (filter(client) &&
        (client->conState() == ReflectorClient::STATE_CONNECTED))
    {
      client->sendMsg(msg);
    }
  }
} /* Reflector::broadcastMsg */


bool Reflector::sendUdpDatagram(ReflectorClient *client, const void *buf,
                                size_t count)
{
  return m_udp_sock->write(client->remoteHost(), client->remoteUdpPort(), buf,
                           count);
} /* Reflector::sendUdpDatagram */


void Reflector::broadcastUdpMsg(const ReflectorUdpMsg& msg,
                                const ReflectorClient::Filter& filter)
{
  for (const auto& item : m_client_con_map)
  {
    ReflectorClient *client = item.second;
    if (filter(client) &&
        (client->conState() == ReflectorClient::STATE_CONNECTED))
    {
      client->sendUdpMsg(msg);
    }
  }
} /* Reflector::broadcastUdpMsg */


void Reflector::requestQsy(ReflectorClient *client, uint32_t tg)
{
  uint32_t current_tg = TGHandler::instance()->TGForClient(client);
  if (current_tg == 0)
  {
    std::cout << client->callsign()
              << ": Cannot request QSY from TG #0" << std::endl;
    return;
  }

  if (tg == 0)
  {
    tg = nextRandomQsyTg();
    if (tg == 0) { return; }
  }

  cout << client->callsign() << ": Requesting QSY from TG #"
       << current_tg << " to TG #" << tg << endl;

  broadcastMsg(MsgRequestQsy(tg),
      ReflectorClient::mkAndFilter(
        v2_client_filter,
        ReflectorClient::TgFilter(current_tg)));
} /* Reflector::requestQsy */


/****************************************************************************
 *
 * Protected member functions
 *
 ****************************************************************************/



/****************************************************************************
 *
 * Private member functions
 *
 ****************************************************************************/

void Reflector::clientConnected(Async::FramedTcpConnection *con)
{
  cout << "Client " << con->remoteHost() << ":" << con->remotePort()
       << " connected" << endl;
  m_client_con_map[con] = new ReflectorClient(this, con, m_cfg);
} /* Reflector::clientConnected */


void Reflector::clientDisconnected(Async::FramedTcpConnection *con,
                           Async::FramedTcpConnection::DisconnectReason reason)
{
  ReflectorClientConMap::iterator it = m_client_con_map.find(con);
  assert(it != m_client_con_map.end());
  ReflectorClient *client = (*it).second;

  TGHandler::instance()->removeClient(client);

  if (!client->callsign().empty())
  {
    cout << client->callsign() << ": ";
  }
  else
  {
    cout << "Client " << con->remoteHost() << ":" << con->remotePort() << " ";
  }
  cout << "disconnected: " << TcpConnection::disconnectReasonStr(reason)
       << endl;

  m_client_con_map.erase(it);

  if (!client->callsign().empty())
  {
    broadcastMsg(MsgNodeLeft(client->callsign()),
        ReflectorClient::ExceptFilter(client));
  }
  Application::app().runTask([=]{ delete client; });
} /* Reflector::clientDisconnected */

std::vector<opus_int16> decodeOpusData(const std::vector<uint8_t>& opusData, int frame_size) {
    // Initialize the Opus decoder
    opus::Decoder decoder(16000, 1);
    if (!decoder.valid()) {
        std::cerr << "Failed to initialize Opus decoder." << std::endl;
        return {};
    }

    // Decode the Opus data to PCM format
    auto pcmData = decoder.Decode({opusData}, frame_size, false);
    if (pcmData.empty()) {
        std::cerr << "Decoding failed or no data was decoded." << std::endl;
        return {};
    }

    return pcmData;
}

std::vector<float> convertPcmToFloat(const std::vector<short>& pcmData) {
    std::vector<float> floatData(pcmData.size());
    for (size_t i = 0; i < pcmData.size(); ++i) {
        // Normalize the PCM data to the range [-1.0, 1.0]
        floatData[i] = pcmData[i] / 32768.0f;
    }
    return floatData;
}

void Reflector::broadcastIfCurrentTalker(ReflectorClient* client, uint32_t tg, const MsgUdpAudio& msg) {
    ReflectorClient *talker = TGHandler::instance()->talkerForTG(tg);
    if (talker == 0) { // If there's no current talker for the TG
        TGHandler::instance()->setTalkerForTG(tg, client); // Set the current client as the talker
        talker = TGHandler::instance()->talkerForTG(tg);
    }
    if (talker == client) { // Ensure the current client is the talker before broadcasting
        TGHandler::instance()->setTalkerForTG(tg, client);
        broadcastUdpMsg(msg,
                        ReflectorClient::mkAndFilter(
                                ReflectorClient::ExceptFilter(client),
                                ReflectorClient::TgFilter(tg)));
    }
}

void Reflector::resetVadStates() {
    processedSamples = 0;
    pcmSampleBuffer.clear(); // Clear any accumulated samples
    preVoiceBuffer.clear(); // Clear the pre-voice buffer
}

void Reflector::udpDatagramReceived(const IpAddress& addr, uint16_t port,
                                    void *buf, int count)
{
  stringstream ss;
  ss.write(reinterpret_cast<const char *>(buf), count);

  ReflectorUdpMsg header;
  if (!header.unpack(ss))
  {
    cout << "*** WARNING: Unpacking message header failed for UDP datagram "
            "from " << addr << ":" << port << endl;
    return;
  }

  ReflectorClient *client = ReflectorClient::lookup(header.clientId());
  if (client == nullptr)
  {
    cerr << "*** WARNING: Incoming UDP datagram from " << addr << ":" << port
         << " has invalid client id " << header.clientId() << endl;
    return;
  }
  if (addr != client->remoteHost())
  {
    cerr << "*** WARNING[" << client->callsign()
         << "]: Incoming UDP packet has the wrong source ip, "
         << addr << " instead of " << client->remoteHost() << endl;
    return;
  }
  if (client->remoteUdpPort() == 0)
  {
    client->setRemoteUdpPort(port);
    client->sendUdpMsg(MsgUdpHeartbeat());
  }
  else if (port != client->remoteUdpPort())
  {
    cerr << "*** WARNING[" << client->callsign()
         << "]: Incoming UDP packet has the wrong source UDP "
            "port number, " << port << " instead of "
         << client->remoteUdpPort() << endl;
    return;
  }

    // Check sequence number
  uint16_t udp_rx_seq_diff = header.sequenceNum() - client->nextUdpRxSeq();
  if (udp_rx_seq_diff > 0x7fff) // Frame out of sequence (ignore)
  {
    cout << client->callsign()
         << ": Dropping out of sequence frame with seq="
         << header.sequenceNum() << ". Expected seq="
         << client->nextUdpRxSeq() << endl;
    return;
  }
  else if (udp_rx_seq_diff > 0) // Frame(s) lost
  {
    cout << client->callsign()
         << ": UDP frame(s) lost. Expected seq=" << client->nextUdpRxSeq()
         << ". Received seq=" << header.sequenceNum() << endl;
  }

  client->udpMsgReceived(header);

  switch (header.type())
  {
    case MsgUdpHeartbeat::TYPE:
      break;

    case MsgUdpAudio::TYPE:
    {
      if (!client->isBlocked())
      {
        MsgUdpAudio msg;
        if (!msg.unpack(ss))
        {
          cerr << "*** WARNING[" << client->callsign()
               << "]: Could not unpack incoming MsgUdpAudioV1 message" << endl;
          return;
        }
        uint32_t tg = TGHandler::instance()->TGForClient(client);
        if (!msg.audioData().empty() && (tg > 0))
        {
            // Only enter this block if VAD is enabled, the callsign is in the list, and voice has not been detected yet.
            if (isVadEnabled && vadEnabledCallsigns.find(client->callsign()) != vadEnabledCallsigns.end()) {

                if(processedSamples < vadGateSampleSize && !client->voiceDetected) {
                    // Always append incoming audio data to the preVoiceBuffer to not lose any data until voice is detected
                    preVoiceBuffer.push_back(msg);

                    // Decode the Opus data to PCM format immediately
                    auto pcmData = decodeOpusData(msg.audioData(), 320); // Assuming frame_size is 320
                    if (pcmData.empty()) {
                        std::cerr << "Decoding failed or returned no data. Skipping this audio data.\n";
                        return; // Skip further processing for this data chunk
                    }

                    auto pcmDataFloat = convertPcmToFloat(pcmData);

                    // Accumulate floating-point PCM samples for processing
                    pcmSampleBuffer.insert(pcmSampleBuffer.end(), pcmDataFloat.begin(), pcmDataFloat.end());
                    int silenceSamples = (16000 * startSilenceMs) / 1000;

                    while (pcmSampleBuffer.size() >= sampleBufferSize && processedSamples < vadGateSampleSize && !client->voiceDetected)
                    {
                        std::vector<float> batchToProcess(pcmSampleBuffer.begin(), pcmSampleBuffer.begin() + static_cast<std::vector<float>::difference_type>(sampleBufferSize));
                        std::fill_n(batchToProcess.begin(), std::min(static_cast<size_t>(silenceSamples), batchToProcess.size()), 0.0f); // Replace the first ms of the buffer with silence
                        vadIterator->process(batchToProcess);
                        processedSamples += sampleBufferSize;

                        if (vadIterator->isVoicePresent())
                        {
                            std::cout << "Voice detected, end\n";
                            client->voiceDetected = true;

                            // Broadcast the pre-voice buffer here
                            if (!preVoiceBuffer.empty())
                            {
                                for (auto& bufferedMsg : preVoiceBuffer) {
                                    // Process or broadcast each MsgUdpAudio object
                                    broadcastIfCurrentTalker(client, tg, bufferedMsg);
                                }
                                preVoiceBuffer.clear();
                                msg.audioData().clear();
                            }
                            break;
                        }

                        pcmSampleBuffer.erase(pcmSampleBuffer.begin(), pcmSampleBuffer.begin() + static_cast<std::vector<float>::difference_type>(sampleBufferSize));
                    }

                    break; // Skip further processing for this data chunk

                } else if (client->voiceDetected) {
                    // If voice has been detected, broadcast the audio data immediately
                    broadcastIfCurrentTalker(client, tg, msg);
                }


            } else {
                // If the callsign is not in the list or VAD is not enabled, broadcast the audio data immediately
                broadcastIfCurrentTalker(client, tg, msg);
            }
        }
          // If the processed samples exceed the VAD gate sample size and voice has not been detected, disconnect the client
          if (processedSamples >= vadGateSampleSize && !client->voiceDetected) {
              client->disconnect();
              resetVadStates();
              return;
          }

      }
      break;
    }

    //case MsgUdpAudio::TYPE:
    //{
    //  if (!client->isBlocked())
    //  {
    //    MsgUdpAudio msg;
    //    if (!msg.unpack(ss))
    //    {
    //      cerr << "*** WARNING[" << client->callsign()
    //           << "]: Could not unpack incoming MsgUdpAudio message" << endl;
    //      return;
    //    }
    //    if (!msg.audioData().empty())
    //    {
    //      if (m_talker == 0)
    //      {
    //        setTalker(client);
    //        cout << m_talker->callsign() << ": Talker start on TG #"
    //             << msg.tg() << endl;
    //      }
    //      if (m_talker == client)
    //      {
    //        gettimeofday(&m_last_talker_timestamp, NULL);
    //        broadcastUdpMsgExcept(tg, client, msg,
    //            ProtoVerRange(ProtoVer(2, 0), ProtoVer::max()));
    //        MsgUdpAudioV1 msg_v1(msg.audioData());
    //        broadcastUdpMsgExcept(tg, client, msg_v1,
    //            ProtoVerRange(ProtoVer(0, 6),
    //                          ProtoVer(1, ProtoVer::max().minor())));
    //      }
    //    }
    //  }
    //  break;
    //}

    case MsgUdpFlushSamples::TYPE:
    {
      uint32_t tg = TGHandler::instance()->TGForClient(client);
      ReflectorClient* talker = TGHandler::instance()->talkerForTG(tg);
      if ((tg > 0) && (client == talker))
      {
        TGHandler::instance()->setTalkerForTG(tg, 0);
      }
        // To be 100% correct the reflector should wait for all connected
        // clients to send a MsgUdpAllSamplesFlushed message but that will
        // probably lead to problems, especially on reflectors with many
        // clients. We therefore acknowledge the flush immediately here to
        // the client who sent the flush request.
      client->sendUdpMsg(MsgUdpAllSamplesFlushed());
      break;
    }

    case MsgUdpAllSamplesFlushed::TYPE:
      // Ignore
      break;

    case MsgUdpSignalStrengthValues::TYPE:
    {
      if (!client->isBlocked())
      {
        MsgUdpSignalStrengthValues msg;
        if (!msg.unpack(ss))
        {
          cerr << "*** WARNING[" << client->callsign()
               << "]: Could not unpack incoming "
                  "MsgUdpSignalStrengthValues message" << endl;
          return;
        }
        typedef MsgUdpSignalStrengthValues::Rxs::const_iterator RxsIter;
        for (RxsIter it = msg.rxs().begin(); it != msg.rxs().end(); ++it)
        {
          const MsgUdpSignalStrengthValues::Rx& rx = *it;
          //std::cout << "### MsgUdpSignalStrengthValues:"
          //  << " id=" << rx.id()
          //  << " siglev=" << rx.siglev()
          //  << " enabled=" << rx.enabled()
          //  << " sql_open=" << rx.sqlOpen()
          //  << " active=" << rx.active()
          //  << std::endl;
          client->setRxSiglev(rx.id(), rx.siglev());
          client->setRxEnabled(rx.id(), rx.enabled());
          client->setRxSqlOpen(rx.id(), rx.sqlOpen());
          client->setRxActive(rx.id(), rx.active());
        }
      }
      break;
    }

    default:
      // Better ignoring unknown messages to make it easier to add messages to
      // the protocol but still be backwards compatible

      //cerr << "*** WARNING[" << client->callsign()
      //     << "]: Unknown UDP protocol message received: msg_type="
      //     << header.type() << endl;
      break;
  }
} /* Reflector::udpDatagramReceived */


void Reflector::onTalkerUpdated(uint32_t tg, ReflectorClient* old_talker,
                                ReflectorClient *new_talker)
{
  if (old_talker != 0)
  {
      resetVadStates();
      old_talker->voiceDetected = false;
    cout << old_talker->callsign() << ": Talker stop on TG #" << tg << endl;
    broadcastMsg(MsgTalkerStop(tg, old_talker->callsign()),
        ReflectorClient::mkAndFilter(
          v2_client_filter,
          ReflectorClient::mkOrFilter(
            ReflectorClient::TgFilter(tg),
            ReflectorClient::TgMonitorFilter(tg))));
    if (tg == tgForV1Clients())
    {
      broadcastMsg(MsgTalkerStopV1(old_talker->callsign()), v1_client_filter);
    }
    broadcastUdpMsg(MsgUdpFlushSamples(),
          ReflectorClient::mkAndFilter(
            ReflectorClient::TgFilter(tg),
            ReflectorClient::ExceptFilter(old_talker)));
  }
  if (new_talker != 0)
  {
    cout << new_talker->callsign() << ": Talker start on TG #" << tg << endl;
    broadcastMsg(MsgTalkerStart(tg, new_talker->callsign()),
        ReflectorClient::mkAndFilter(
          v2_client_filter,
          ReflectorClient::mkOrFilter(
            ReflectorClient::TgFilter(tg),
            ReflectorClient::TgMonitorFilter(tg))));
    if (tg == tgForV1Clients())
    {
      broadcastMsg(MsgTalkerStartV1(new_talker->callsign()), v1_client_filter);
    }
  }
} /* Reflector::setTalker */


void Reflector::httpRequestReceived(Async::HttpServerConnection *con,
                                    Async::HttpServerConnection::Request& req)
{
  //std::cout << "### " << req.method << " " << req.target << std::endl;

  Async::HttpServerConnection::Response res;
  if ((req.method != "GET") && (req.method != "HEAD"))
  {
    res.setCode(501);
    res.setContent("application/json",
        "{\"msg\":\"" + req.method + ": Method not implemented\"}");
    con->write(res);
    return;
  }

  if (req.target != "/status")
  {
    res.setCode(404);
    res.setContent("application/json",
        "{\"msg\":\"Not found!\"}");
    con->write(res);
    return;
  }

  Json::Value status;
  status["nodes"] = Json::Value(Json::objectValue);
  for (const auto& item : m_client_con_map)
  {
    ReflectorClient* client = item.second;
    Json::Value node(client->nodeInfo());
    //node["addr"] = client->remoteHost().toString();
    node["protoVer"]["majorVer"] = client->protoVer().majorVer();
    node["protoVer"]["minorVer"] = client->protoVer().minorVer();
    auto tg = client->currentTG();
    if (!TGHandler::instance()->showActivity(tg))
    {
      tg = 0;
    }
    node["tg"] = tg;
    node["restrictedTG"] = TGHandler::instance()->isRestricted(tg);
    Json::Value tgs = Json::Value(Json::arrayValue);
    const std::set<uint32_t>& monitored_tgs = client->monitoredTGs();
    for (std::set<uint32_t>::const_iterator mtg_it=monitored_tgs.begin();
         mtg_it!=monitored_tgs.end(); ++mtg_it)
    {
      tgs.append(*mtg_it);
    }
    node["monitoredTGs"] = tgs;
    bool is_talker = TGHandler::instance()->talkerForTG(tg) == client;
    node["isTalker"] = is_talker;

    if (node.isMember("qth") && node["qth"].isArray())
    {
      //std::cout << "### Found qth" << std::endl;
      Json::Value& qths(node["qth"]);
      for (Json::Value::ArrayIndex i=0; i<qths.size(); ++i)
      {
        Json::Value& qth(qths[i]);
        if (qth.isMember("rx") && qth["rx"].isObject())
        {
          //std::cout << "### Found rx" << std::endl;
          Json::Value::Members rxs(qth["rx"].getMemberNames());
          for (Json::Value::Members::const_iterator it=rxs.begin(); it!=rxs.end(); ++it)
          {
            //std::cout << "### member=" << *it << std::endl;
            const std::string& rx_id_str(*it);
            if (rx_id_str.size() == 1)
            {
              char rx_id(rx_id_str[0]);
              Json::Value& rx(qth["rx"][rx_id_str]);
              if (client->rxExist(rx_id))
              {
                rx["siglev"] = client->rxSiglev(rx_id);
                rx["enabled"] = client->rxEnabled(rx_id);
                rx["sql_open"] = client->rxSqlOpen(rx_id);
                rx["active"] = client->rxActive(rx_id);
              }
            }
          }
        }
        if (qth.isMember("tx") && qth["tx"].isObject())
        {
          //std::cout << "### Found tx" << std::endl;
          Json::Value::Members txs(qth["tx"].getMemberNames());
          for (Json::Value::Members::const_iterator it=txs.begin(); it!=txs.end(); ++it)
          {
            //std::cout << "### member=" << *it << std::endl;
            const std::string& tx_id_str(*it);
            if (tx_id_str.size() == 1)
            {
              char tx_id(tx_id_str[0]);
              Json::Value& tx(qth["tx"][tx_id_str]);
              if (client->txExist(tx_id))
              {
                tx["transmit"] = client->txTransmit(tx_id);
              }
            }
          }
        }
      }
    }
    status["nodes"][client->callsign()] = node;
  }
  std::ostringstream os;
  Json::StreamWriterBuilder builder;
  builder["commentStyle"] = "None";
  builder["indentation"] = ""; //The JSON document is written on a single line
  Json::StreamWriter* writer = builder.newStreamWriter();
  writer->write(status, &os);
  delete writer;
  res.setContent("application/json", os.str());
  if (req.method == "HEAD")
  {
    res.setSendContent(false);
  }
  res.setCode(200);
  con->write(res);
} /* Reflector::requestReceived */


void Reflector::httpClientConnected(Async::HttpServerConnection *con)
{
  //std::cout << "### HTTP Client connected: "
  //          << con->remoteHost() << ":" << con->remotePort() << std::endl;
  con->requestReceived.connect(sigc::mem_fun(*this, &Reflector::httpRequestReceived));
} /* Reflector::httpClientConnected */


void Reflector::httpClientDisconnected(Async::HttpServerConnection *con,
    Async::HttpServerConnection::DisconnectReason reason)
{
  //std::cout << "### HTTP Client disconnected: "
  //          << con->remoteHost() << ":" << con->remotePort()
  //          << ": " << Async::HttpServerConnection::disconnectReasonStr(reason)
  //          << std::endl;
} /* Reflector::httpClientDisconnected */


void Reflector::onRequestAutoQsy(uint32_t from_tg)
{
  uint32_t tg = nextRandomQsyTg();
  if (tg == 0) { return; }

  std::cout << "Requesting auto-QSY from TG #" << from_tg
            << " to TG #" << tg << std::endl;

  broadcastMsg(MsgRequestQsy(tg),
      ReflectorClient::mkAndFilter(
        v2_client_filter,
        ReflectorClient::TgFilter(from_tg)));
} /* Reflector::onRequestAutoQsy */


uint32_t Reflector::nextRandomQsyTg(void)
{
  if (m_random_qsy_tg == 0)
  {
    std::cout << "*** WARNING: QSY request for random TG "
              << "requested but RANDOM_QSY_RANGE is empty" << std::endl;
    return 0;
  }

  assert (m_random_qsy_tg != 0);
  uint32_t range_size = m_random_qsy_hi-m_random_qsy_lo+1;
  uint32_t i;
  for (i=0; i<range_size; ++i)
  {
    m_random_qsy_tg = (m_random_qsy_tg < m_random_qsy_hi) ?
      m_random_qsy_tg+1 : m_random_qsy_lo;
    if (TGHandler::instance()->clientsForTG(m_random_qsy_tg).empty())
    {
      return m_random_qsy_tg;
    }
  }

  std::cout << "*** WARNING: No random TG available for QSY" << std::endl;
  return 0;
} /* Reflector::nextRandomQsyTg */


void Reflector::ctrlPtyDataReceived(const void *buf, size_t count)
{
  const char* ptr = reinterpret_cast<const char*>(buf);
  const std::string cmdline(ptr, ptr + count);
  //std::cout << "### Reflector::ctrlPtyDataReceived: " << cmdline
  //          << std::endl;
  std::istringstream ss(cmdline);
  std::ostringstream errss;
  std::string cmd;
  if (!(ss >> cmd))
  {
    errss << "Invalid PTY command '" << cmdline << "'";
    goto write_status;
  }

  if (cmd == "CFG")
  {
    std::string section, tag, value;
    if (!(ss >> section >> tag >> value) || !ss.eof())
    {
      errss << "Invalid PTY command '" << cmdline << "'. "
               "Usage: CFG <section> <tag> <value>";
      goto write_status;
    }
    m_cfg->setValue(section, tag, value);
  }
  else
  {
    errss << "Unknown PTY command '" << cmdline
          << "'. Valid commands are: CFG";
  }

  write_status:
    if (!errss.str().empty())
    {
      std::cerr << "*** ERROR: " << errss.str() << std::endl;
      m_cmd_pty->write(std::string("ERR:") + errss.str() + "\n");
      return;
    }
    m_cmd_pty->write("OK\n");
} /* Reflector::ctrlPtyDataReceived */


void Reflector::cfgUpdated(const std::string& section, const std::string& tag)
{
  std::string value;
  if (!m_cfg->getValue(section, tag, value))
  {
    std::cout << "*** ERROR: Failed to read updated configuration variable '"
              << section << "/" << tag << "'" << std::endl;
    return;
  }

  if (section == "GLOBAL")
  {
    if (tag == "SQL_TIMEOUT_BLOCKTIME")
    {
      unsigned t = TGHandler::instance()->sqlTimeoutBlocktime();
      if (!SvxLink::setValueFromString(t, value))
      {
        std::cout << "*** ERROR: Failed to set updated configuration "
                     "variable '" << section << "/" << tag << "'" << std::endl;
        return;
      }
      TGHandler::instance()->setSqlTimeoutBlocktime(t);
      //std::cout << "### New value for " << tag << "=" << t << std::endl;
    }
    else if (tag == "SQL_TIMEOUT")
    {
      unsigned t = TGHandler::instance()->sqlTimeout();
      if (!SvxLink::setValueFromString(t, value))
      {
        std::cout << "*** ERROR: Failed to set updated configuration "
                     "variable '" << section << "/" << tag << "'" << std::endl;
        return;
      }
      TGHandler::instance()->setSqlTimeout(t);
      //std::cout << "### New value for " << tag << "=" << t << std::endl;
    }
  }
} /* Reflector::cfgUpdated */


/*
 * This file has not been truncated
 */

