// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include <sstream>

#include "ToJson.h"
#include "ApiData.h"

// Full-state save/restore blob format version ("PSV1"). The blob layout is a
// raw memory image of State plus Game bookkeeping, so it is only readable by
// the exact build that wrote it — bump this on ANY engine/layout change.
static inline constexpr int kStateBlobMagic = 0x32565350;

struct StartData {
	ApiData* battlePtr;
	int errorPlayer;
	int errorType;
};

struct SerialData {
	const char8_t* json;
	const char* data;
	int count;
	int selectPlayer;
};

inline StartData ApiBattleStart(int* cards, uint32_t forcedSeed = 0, bool useDeviceRand = true) {
	ApiData* data = new ApiData();
	data->apiDataType = 1;

	std::random_device rd;
	GameConfig config = {};
	config.seed = forcedSeed ? forcedSeed : rd();
	config.recordLog = true;
	config.deviceRand = useDeviceRand;
	for (int i = 0; i < 2; i++) {
		std::unordered_map<std::u8string, int> nameCount;
		bool aceSpec = false;
		bool basic = false;
		for (int j = 0; j < DECK_SIZE; j++) {
			CardId id = cards[i * DECK_SIZE + j];
			if (!CardTable.contains(id)) {
				delete data;
				return { nullptr, i, 1 };
			}

			const CardMaster& master = CardTable.at(id);
			if (master.aceSpec) {
				if(aceSpec){
					delete data;
					return { nullptr, i, 4 };
				} else {
					aceSpec = true;
				}
			}

			if (master.cardType == CardType::Pokemon && master.evolutionType == EvolutionType::Basic) {
				basic = true;
			}

			int& count = nameCount[master.name];
			count++;
			if (count > DECK_SAME_CARD_MAX) {
				if (master.cardType != CardType::BasicEnergy) {
					delete data;
					return { nullptr, i, 2 };
				}
			}

			config.decks[i].cards[j] = cards[i * DECK_SIZE + j];
		}
		if (!basic) {
			delete data;
			return { nullptr, i, 3 };
		}
	}

	data->init(config);
	if (forcedSeed == 0) {
		// Stock behavior: re-seed from hardware entropy. A forced seed instead
		// keeps Game::init's rng = mt19937(seed) for deterministic battles.
		std::seed_seq seq{ rd(), rd(), rd(), rd() };
		data->game.rng = std::mt19937(seq);
	}
	data->start();
	data->next();
	return { data, -1, 0 };
}

// Serialize the FULL battle state (no erasePlayerData fog pass) plus the Game
// bookkeeping that lives outside State: GameConfig (minus deckNames — only
// used for client-side naming), actionCount, remainingTime, and the mt19937
// engine (canonical textual form, exact round-trip). startTime is skipped:
// it only matters mid-timer with config.timeLimit > 0.
inline SerialData ApiSaveState(ApiData* data) {
	BinaryWriter& b = data->writer;
	b.clear();
	b.set(kStateBlobMagic);
	data->state.serialize(b);

	const GameConfig& c = data->game.config;
	b.set(&c.decks, sizeof(c.decks));
	b.set(&c.seed, (const uint8_t*)(&c + 1) - (const uint8_t*)&c.seed);
	b.set(&data->game.actionCount, sizeof(data->game.actionCount));
	b.set(&data->game.remainingTime, sizeof(data->game.remainingTime));

	std::ostringstream oss;
	oss << data->game.rng;
	const std::string rngText = oss.str();
	b.set((int)rngText.size());
	b.set(rngText.data(), rngText.size());

	b.set(data->selected);

	b.toBase64();
	return { nullptr, b.base64.data(), (int)b.base64.size(), data->state.selectPlayer };
}

// Mirror of ApiSaveState: build a live battle (apiDataType 1) from a blob.
// The saved GameConfig is restored verbatim, deviceRand included — restoring
// a deterministic future requires the blob to come from a seeded battle.
inline StartData ApiBattleStartFromState(const char* base64, int count) {
	ApiData* data = new ApiData();
	data->apiDataType = 1;

	BinaryReader& r = data->reader;
	r.base64.assign(base64, base64 + count);
	r.fromBase64();
	if ((int)r.buf.size() < (int)sizeof(int) || r.readInt() != kStateBlobMagic) {
		delete data;
		return { nullptr, -1, 100 };
	}
	data->state.clear();
	data->state.deserialize(r);
	data->state.game = &data->game;

	GameConfig& c = data->game.config;
	r.set(&c.decks, sizeof(c.decks));
	r.set(&c.seed, (uint8_t*)(&c + 1) - (uint8_t*)&c.seed);
	r.set(&data->game.actionCount, sizeof(data->game.actionCount));
	r.set(&data->game.remainingTime, sizeof(data->game.remainingTime));

	std::string rngText((size_t)r.readInt(), '\0');
	r.set(rngText.data(), rngText.size());
	std::istringstream iss(rngText);
	iss >> data->game.rng;

	r.set(data->selected);

	// selectCount=0 / preGetSelectCount=-1 defaults make the first
	// GetBattleData emit a fresh vis snapshot; Select/next() resume from the
	// deserialized functionStack.
	return { data, -1, 0 };
}

inline ApiData* ApiAgentStart() {
	ApiData* data = new ApiData();
	data->apiDataType = 2;
	GameConfig& config = data->game.config;
	config.seed = std::random_device()();
	config.recordLog = true;
	data->game.rng = std::mt19937(config.seed);
	data->state.game = &data->game;
	return data;
}

inline void ApiBattleFinish(ApiData* data) {
	delete data;
}

inline SerialData ApiGetBattleData(ApiData* data) {
	BinaryWriter& b = data->writer;
	State state;
	state.clear();
	state = data->state;
	state.erasePlayerData(state.selectPlayer);
	b.clear();
	state.serialize(b);
	b.toBase64();

	data->jsonBuilder.clear();
	ToJsonApi(data->state, data->jsonBuilder, data->state.nextLogStart());

	return { data->jsonBuilder.buf.c_str(), b.base64.data(), (int)b.base64.size(), state.selectPlayer };
}

inline void SetBattleData(ApiData* data, const char* base64, int count) {
	data->reader.base64.resize(count);
	std::copy(base64, base64 + count, data->reader.base64.begin());
	data->reader.fromBase64();
	data->state.deserialize(data->reader);
}

inline int ApiSelect(ApiData* data, int* select, int selectCount) {
	State& state = data->state;
	state.selected.clear();
	for (int i : range(selectCount)) {
		state.selected.push_back(select[i]);
	}
	int error = state.checkPlayerSelect();
	if (error) {
		return error;
	}
	data->selected = state.selected;

	data->next(); 
	while (!state.isFinish() && state.selectMax == 0) {
		state.selected.clear();
		data->next();
	}
	data->selectCount++;
	return 0;
}

inline void ApiVisualizeData(JsonBuilder& j, const std::vector<std::u8string>& visData) {
	j.clear();
	j.append('[');
	for (int i : range(visData)) {
		j.comma(i);
		j.buf += visData[i];
	}
	j.append(']');
}

inline void SearchReturnJson(JsonBuilder& j, const SearchInfo& si) {
	j.clear();
	ToJsonSearch(si.state, j, si.searchId, si.errorCode);
}

inline SearchInfo ApiSearchBegin(ApiData* data, const SearchStartConfig& config) {
	return data->search.start(config, data->state);
}

inline SearchInfo ApiSearchStep(ApiData* data, long long searchId, int* select, int selectCount) {
	data->selected.clear();
	for (int i : range(selectCount)) {
		data->selected.push_back(select[i]);
	}
	return data->search.step(searchId, data->selected);
}

inline void ApiSearchEnd(ApiData* data) {
	data->search.clear();
}

inline int ApiSearchRelease(ApiData* data, long long searchId) {
	return data->search.clearSingle(searchId);
}


inline std::u8string EscapeJsonText(const std::u8string& text) {
	std::u8string result;
	for (char8_t c : text) {
		if (c == '\n') {
			result += u8"\\n";
		} else if (c == '"') {
			result += u8"\\\"";
		} else {
			result += c;
		}
	}
	return result;
}

inline void SkillJson(JsonBuilder& j, const Skill& skill) {
	j.append('{');
	j.appendKeyValue("name", skill.nameEn);
	j.appendCommaKeyValue("text", EscapeJsonText(skill.textEn));
	j.append('}');
}

inline int EnergyWeakness(EnergyType type) {
	if (type == EnergyType::Colorless) {
		return -1;
	} else {
		return EnergyTypeIndex(type);
	}
}

inline void ApiAllCard(JsonBuilder& j){
	j.clear();
	j.append('[');

	std::vector<const CardMaster*> cards;
	for (const auto& entry : CardTable) {
		cards.push_back(&entry.second);
	}
	std::sort(cards.begin(), cards.end(), [](const CardMaster* left, const CardMaster* right) {
		return left->cardId < right->cardId;
	});

	for (int i : range(cards)) {
		const CardMaster& card = *cards[i];
		j.comma(i);
		j.append('{');

		j.appendKeyValue("cardId", card.cardId);
		j.appendCommaKeyValue("name", card.nameEn);
		j.appendCommaKeyValue("cardType", (int)card.cardType);
		j.appendCommaKeyValue("pokemonType", (int)card.pokemonType);
		j.appendCommaKeyValue("evolutionType", (int)card.evolutionType);
		j.appendCommaKeyValue("retreatCost", card.retreatCost);
		j.appendCommaKeyValue("hp", card.hp);
		j.appendCommaKeyValueOrNull("weakness", EnergyWeakness(card.weakness));
		j.appendCommaKeyValueOrNull("resistance", EnergyWeakness(card.resistance));
		EnergyType et = card.energyType;
		if (et == (EnergyType::Psychic | EnergyType::Darkness)) {
			et = EnergyType::Colorless;
		}
		j.appendCommaKeyValue("energyType", EnergyTypeIndex(et));
		j.appendCommaKeyValue("basic", card.evolutionType == EvolutionType::Basic);
		j.appendCommaKeyValue("stage1", card.evolutionType == EvolutionType::Stage1);
		j.appendCommaKeyValue("stage2", card.evolutionType == EvolutionType::Stage2);
		j.appendCommaKeyValue("ex", card.pokemonType == PokemonType::Ex);
		j.appendCommaKeyValue("megaEx", card.pokemonType == PokemonType::MegaEx);
		j.appendCommaKeyValue("tera", card.tera);
		j.appendCommaKeyValue("aceSpec", card.aceSpec);
		if (card.evolvesFrom == u8"") {
			j.appendCommaKeyValueOrNull("evolvesFrom", u8"");
		} else {
			const CardMaster& evolve = CardTable.at(NameTable.at(card.evolvesFrom));
			j.appendCommaKeyValueOrNull("evolvesFrom", evolve.nameEn);
		}

		j.appendCommaKey("skills");
		j.append('[');
		auto skills = card.getSkills();
		for (int n : range(skills)) {
			j.comma(n);
			SkillJson(j, *skills[n]);
		}
		j.append(']');

		j.appendCommaKey("attacks");
		j.append('[');
		for (int n : range(card.attacks)) {
			j.comma(n);
			j.append(card.attacks[n]->attackId);
		}
		j.append(']');

		j.append('}');
	}

	j.append(']');
}

inline void ApiAllAttack(JsonBuilder& j) {
	j.clear();
	j.append('[');

	std::vector<const Attack*> attacks;
	for (const auto& entry : AttackTable) {
		attacks.push_back(&entry.second);
	}
	std::sort(attacks.begin(), attacks.end(), [](const Attack* left, const Attack* right) {
		return left->attackId < right->attackId;
	});

	for (int i : range(attacks)) {
		const Attack& attack = *attacks[i];
		j.comma(i);
		j.append('{');

		j.appendKeyValue("attackId", attack.attackId);
		j.appendCommaKeyValue("name", attack.nameEn);
		j.appendCommaKeyValue("text", EscapeJsonText(attack.textEn));
		j.appendCommaKeyValue("damage", attack.damage);

		j.appendCommaKey("energies");
		j.append('[');
		for (int n : range(attack.energies)) {
			j.comma(n);
			j.append(EnergyTypeIndex(attack.energies[n]));
		}
		j.append(']');

		j.append('}');
	}

	j.append(']');
}


