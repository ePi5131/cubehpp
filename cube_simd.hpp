// cube_simd.hpp
// (C) 2023 ePi

#pragma once

#include <cube.hpp>
#include <intrin.h>

#include <thread>
#include <vector>

#include <ppl.h>

#include <simden.hpp>

namespace CubeHpp {
	using AVX2 = simden::intrinsics<simden::intrinsics_flag(
		simden::itSSE,
		simden::itSSE2,
		simden::itSSE3,
		simden::itSSSE3,
		simden::itSSE41,
		simden::itSSE42,
		simden::itAVX,
		simden::itFMA,
		simden::itAVX2
	)>;

    class cube1d_bgra32_avx2 : public cube1d_bgra32 {
		struct PreCalc {
			AVX2::f32x8 afts;
			AVX2::f32x8 aftt;
			AVX2::f32x8 resolution;
			AVX2::i32x8 sizem1;
		};

	public:
		cube1d_bgra32_avx2(cube_base&& b) :
			cube1d_bgra32{ std::move(b) }
		{}

        constexpr static size_t simd_unit = 2;

        void apply(BGRA32* ptr, const PreCalc& pc) const {
            const auto bgra_f32 = [](BGRA32* ptr) {
				const auto m1 = AVX2::u8x8::load(ptr);
				const auto m2 = AVX2::convert_to<AVX2::f32x8>(m1);
				return m2;
			} (ptr);

			const auto m1 = bgra_f32 * pc.resolution;
			const auto mi1f = AVX2::floor(m1);
			const auto mi1 = AVX2::convert_to<AVX2::i32x8>(mi1f);
			const auto mt = m1 - mi1f;

			const auto data = domains.begin()->data();

			static constinit const AVX2::i32x8 c1{ 1,1,1,1,1,1,1,1 };
			const auto mi2 = AVX2::min_(mi1 + c1, pc.sizem1);

			auto load_domain = [data](const AVX2::i32x8& x) {
				static constinit const AVX2::i32x8 c3{ 3,3,3,3,3,3,3,3 };
				static constinit const AVX2::i32x8 co{ 2,1,0,0,2,1,0,0 };

				return AVX2::gather(data, x * c3 + co);
			};

			const auto p1 = load_domain(mi1);
			const auto p2 = load_domain(mi2);

			static constinit const AVX2::f32x8 c1f{ 1.f ,1.f ,1.f ,1.f ,1.f ,1.f ,1.f ,1.f };
            const auto applied = AVX2::fma(p1, c1f - mt, p2 * mt);

			const auto result = AVX2::fma(bgra_f32, pc.afts, applied * pc.aftt);

            auto f32_to_bgra = [](BGRA32* ptr, const AVX2::f32x8& m) {
                const auto m1 = AVX2::convert_to<AVX2::i32x8>(m);
                static constinit const AVX2::i32x8 c0 = { 0,0,0,0,0,0,0,0 };
                static constinit const AVX2::i32x8 c255 = { 255,255,255,255,255,255,255,255 };
                const auto m2 = AVX2::clamp(m1, c0, c255);
                const auto m3 = AVX2::convert_to<AVX2::u8x8>(m2);
				AVX2::store64(ptr, m3);
            };
            f32_to_bgra(ptr, result);
        }
    
        void apply(BGRA32* d, size_t count, float af) const {
			const auto afs = 1.f - af;
			const auto ssizem1 = static_cast<int>(size) - 1;
			const PreCalc pc{
				.afts = {
					afs,afs,afs,1.f,
					afs,afs,afs,1.f
				},
				.aftt{
					af,af,af,0.f,
					af,af,af,0.f
				},
				.resolution{
					resolution,resolution,resolution,resolution,resolution,resolution,resolution,resolution
				},
				.sizem1{
					ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1
				}
			};

			#ifndef CUBEHPP_PARALLEL_OFF
			concurrency::parallel_for(0u, count, [this, d, &pc](size_t i) {
			#else
			for (size_t i = 0u; i < count; i++) {
			#endif
				apply(d + i * simd_unit, pc);
			#ifndef CUBEHPP_PARALLEL_OFF
			});
			#else
			}
			#endif

        }
    };

	class cube3d_bgra32_avx2 : public cube3d_bgra32 {
		struct PreCalc {
			AVX2::f32x8 afts;
			AVX2::f32x8 aftt;
			AVX2::f32x8 resolution;
			AVX2::i32x8 size;
			AVX2::i32x8 sizem1;
		};


		// x>=y y>=z z>=x
		// 0    0    -    z,y,x
		// 1    0    0    x,z,y
		// 0    1    0    y,x,z
		// 1    1    0    x,y,z
		// 0    0    -    z,y,x
		// 1    0    1    z,x,y
		// -    1    1    y,z,x
		// -    1    1    y,z,x

		static constexpr std::array<int, 4> o1 = { 0,1,1,1 };
		static constexpr std::array<int, 4> o2 = { 0,0,1,1 };
		static constexpr std::array<int, 4> o3 = { 0,0,0,1 };

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> oxt{
			o3, o1, o2, o1, o3, o2, o3, o3
		};

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> oyt{
            o2, o3, o1, o2, o2, o3, o1, o1
		};

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> ozt{
			o1, o2, o3, o3, o1, o1, o2, o2
		};

		static constexpr std::array<int, 4> azyx = { 2,1,0 };
		static constexpr std::array<int, 4> azxy = { 2,0,1 };
		static constexpr std::array<int, 4> ayzx = { 1,2,0 };
		static constexpr std::array<int, 4> ayxz = { 1,0,2 };
		static constexpr std::array<int, 4> axzy = { 0,2,1 };
		static constexpr std::array<int, 4> axyz = { 0,1,2 };

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> apit{
			azyx,
			axzy,
			ayxz,
			axyz,
			azyx,
			azxy,
			ayzx,
			ayzx
		};
	public:
		cube3d_bgra32_avx2(cube_base&& b) :
			cube3d_bgra32{ std::move(b) }
		{
			domains.resize(domains.size() + 2); // padding
		}

        constexpr static size_t simd_unit = 2;

        void apply(BGRA32* ptr, const PreCalc& pc) const {
            auto bgra_to_f32 = [this](BGRA32* ptr) {
                const auto m1 = AVX2::u8x8::load(ptr);
                const auto m2 = AVX2::convert_to<AVX2::f32x8>(m1);
				return AVX2::permute<2, 1, 0, 3>(m2);
            };

            const auto bgra_f32 = bgra_to_f32(ptr);
            const auto m1 = bgra_f32 * pc.resolution;
            const auto midx1f = AVX2::floor(m1);
            const auto midi = AVX2::convert_to<AVX2::i32x8>(midx1f);
            const auto mt = m1 - midx1f;
            
			auto [cmpi0, cmpi1] = [](const AVX2::f32x8& m) {
				// y z x .
				const auto m1 = AVX2::permute<1, 2, 0, 0>(m);

				static constinit const AVX2::i32x8 mask{
					0b001'00, 0b010'00, 0b100'00, 0,
					0b001'00, 0b010'00, 0b100'00, 0,
				}; // { 0b1, 0b10, 0b100, 0 } * 4

				// x>=y y>=z z>=x
                const auto ges = AVX2::cast_to<AVX2::i32x8>(AVX2::cmp_ge(m, m1)) & mask;
                
				const auto m3 = AVX2::permute<0, 0, 0, 0>(ges) | AVX2::permute<1, 1, 1, 1>(ges);
                //return m3 | m4;
				const auto m4 = AVX2::permute<2, 2, 2, 2>(ges) | m3;
				//return m4;
				return std::make_tuple(AVX2::extract<int, 0>(m4), AVX2::extract<int, 4>(m4));
			}(mt);

			auto myloadu2 = [](const int* ptr, int i0, int i1) {
				// require align
				return AVX2::i32x8{
					AVX2::i32x4::load<true>(ptr + i0),
					AVX2::i32x4::load<true>(ptr + i1)
				};
			};

			// 足し算が8つあるけど並列にしないほうが速い
			const auto ox = myloadu2(oxt.data()->data(), cmpi0, cmpi1);
			const auto oy = myloadu2(oyt.data()->data(), cmpi0, cmpi1);
			const auto oz = myloadu2(ozt.data()->data(), cmpi0, cmpi1);
			const auto api = myloadu2(apit.data()->data(), cmpi0, cmpi1);

			//const AVX2::i32x8 tables_m = _mm256_load_si256(reinterpret_cast<const simden::m256i*>(&tables));
			//const auto tables_r = tables_m + cmp_idx;
			//alignas(simden::m256i) std::array<int, 8> tables_a;
			//tables_r.store<true>(tables_a);
			//
			//auto myloadu2 = [](int ptr0, int ptr1) {
			//	// require align
			//	return _mm256_set_m128i(
			//		_mm_loadu_si128(reinterpret_cast<const simden::m128i*>(ptr1)),
			//		_mm_loadu_si128(reinterpret_cast<const simden::m128i*>(ptr0))
			//	);
			//};
			//const auto ox = myloadu2(tables_a[0], tables_a[4]);
			//const auto oy = myloadu2(tables_a[1], tables_a[5]);
			//const auto oz = myloadu2(tables_a[2], tables_a[6]);
			//const auto api = myloadu2(tables_a[3], tables_a[7]);

            const auto ap = [](const AVX2::f32x8& mt, const AVX2::i32x8& api) {
				const auto m1 = AVX2::permutev(mt, api);

				const AVX2::f32x8 m2 = [](const AVX2::f32x8& m) {
					const auto m1 = AVX2::permute<0, 0, 1, 2>(m);
					auto low = AVX2::cast_to<AVX2::f32x4>(m1);
					auto high = AVX2::extract<AVX2::f32x4, 1>(m1);
					low = AVX2::insert<0>(low, 1.f);
					high = AVX2::insert<0>(high, 1.f);
					return AVX2::f32x8{ low, high };
				}(m1);
				
				const AVX2::f32x8 m3 = [](const AVX2::f32x8& m) {
					auto low = AVX2::cast_to<AVX2::f32x4>(m);
					auto high = AVX2::extract<AVX2::f32x4, 1>(m);
					low = AVX2::insert<3>(low, 0.f);
					high = AVX2::insert<3>(high, 0.f);
					return AVX2::f32x8{ low, high };
				}(m1);
				return m2 - m3;
			} (mt, api);

			auto ix = AVX2::permute<0, 0, 0, 0>(midi) + ox;
			auto iy = AVX2::permute<1, 1, 1, 1>(midi) + oy;
			auto iz = AVX2::permute<2, 2, 2, 2>(midi) + oz;
            
			auto ip = [data = domains.cbegin()->data()](const AVX2::i32x8& pi, const AVX2::f32x8& ap) {
				auto load_domain = [data](int i0, int i1) {
					return AVX2::f32x8{
						AVX2::f32x4::load<false>(data + i0),
						AVX2::f32x4::load<false>(data + i1),
					};
				};

				static constinit const AVX2::i32x8 c3{ 3,3,3,3,3,3,3,3 };
				const auto pi3 = pi * c3;
				alignas(AVX2::i32x8) std::array<int, 8> pi3a;
				pi3.store<true>(pi3a);

				const auto m1 = AVX2::fma(load_domain(pi3a[0], pi3a[4]), AVX2::permute<0, 0, 0, 0>(ap), load_domain(pi3a[1], pi3a[5]) * AVX2::permute<1, 1, 1, 1>(ap));
				const auto m2 = AVX2::fma(load_domain(pi3a[2], pi3a[6]), AVX2::permute<2, 2, 2, 2>(ap), load_domain(pi3a[3], pi3a[7]) * AVX2::permute<3, 3, 3, 3>(ap));
				const auto m3 = m1 + m2;
				return m3;
			};

			const auto idx3d_m = [&pc](const AVX2::i32x8& mx, const AVX2::i32x8& my, const AVX2::i32x8& mz) {
				const auto m1 = AVX2::min_(mz, pc.sizem1) * pc.size;
				const auto m2 = m1 + AVX2::min_(my, pc.sizem1);
				const auto m3 = m2 * pc.size;
				const auto m4 = m3 + AVX2::min_(mx, pc.sizem1);
				return m4;
			}(ix, iy, iz);

            const auto applied = ip(idx3d_m, ap);
			const auto result = AVX2::fma(bgra_f32, pc.afts, applied * pc.aftt);

            auto f32_to_bgra = [](BGRA32* ptr, const AVX2::f32x8& m) {
                const auto m1 = AVX2::convert_to<AVX2::i32x8>(AVX2::round(m));
				static constinit const AVX2::i32x8 c0 = { 0,0,0,0,0,0,0,0 };
				static constinit const AVX2::i32x8 c255 = { 255,255,255,255,255,255,255,255 };
                const auto m2 = AVX2::clamp(m1, c0, c255);
				// permute and shuffle
                const auto m3 = [](const AVX2::i32x8& x) {
					#ifdef SIMDEN_EMULATE_INTRINSICS
					const auto m1 = AVX2::permute<2, 1, 0, 3>(x);
					const auto m2 = AVX2::convert_to<AVX2::u8x8>(m1);
					return m2;
					#else
					static constinit const simden::m256i mask_u32_to_permuted_u8{ .m256i_i8{
						8, 4, 0, 12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
						8, 4, 0, 12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
					} };
					const auto m1 = _mm256_shuffle_epi8(x, mask_u32_to_permuted_u8);
					static constinit const simden::m256i mask_permute_compress{ .m256i_i32{
						0,4,0,0,0,0,0,0,
					} };
					const auto m2 = _mm256_permutevar8x32_epi32(m1, mask_permute_compress);
					return m2;
					#endif
				} (m2);
				AVX2::store64(ptr, m3);
            };
            f32_to_bgra(ptr, result);
        }
	
        void apply(BGRA32* d, size_t count, float af) const {
			const auto afs = 1.f - af;
			const auto ssize = static_cast<int>(size);
			const auto ssizem1 = ssize - 1;
			const PreCalc pc{
				.afts{
					afs, afs, afs, 1.f,
					afs, afs, afs, 1.f
				},
				.aftt{
					af, af, af, 0.f,
					af, af, af, 0.f
				},
				.resolution{
					resolution,resolution,resolution,resolution,resolution,resolution,resolution,resolution
				},
				.size{
					ssize,ssize,ssize,ssize,ssize,ssize,ssize,ssize
				},
				.sizem1{
					ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1
				}
			};

			#ifndef CUBEHPP_PARALLEL_OFF
			concurrency::parallel_for(0u, count, [this, d, &pc](size_t i) {
			#else
			for (size_t i = 0; i < count; i++) {
			#endif
				apply(d + i * simd_unit, pc);
			#ifndef CUBEHPP_PARALLEL_OFF
			});
			#else
			}
			#pragma message("parallel off AVX2")
			#endif
        }
    };

	using AVX512 = simden::intrinsics<simden::intrinsics_flag(
		simden::itSSE,
		simden::itSSE2,
		simden::itSSE3,
		simden::itSSSE3,
		simden::itSSE41,
		simden::itSSE42,
		simden::itAVX,
		simden::itFMA,
		simden::itAVX2,
		simden::itAVX512F
	)>;
	
	class cube3d_bgra32_avx512 : public cube3d_bgra32 {
		struct PreCalc {
			AVX512::f32x16 afts;
			AVX512::f32x16 aftt;
			AVX512::f32x16 resolution;
			AVX512::i32x16 size;
			AVX512::i32x16 sizem1;
			AVX512::i32x16 data;
		};

		// x>=y y>=z z>=x
		// 0    0    -    z,y,x
		// 1    0    0    x,z,y
		// 0    1    0    y,x,z
		// 1    1    0    x,y,z
		// 0    0    -    z,y,x
		// 1    0    1    z,x,y
		// -    1    1    y,z,x
		// -    1    1    y,z,x

		static constexpr std::array<int, 4> o1 = { 0,1,1,1 };
		static constexpr std::array<int, 4> o2 = { 0,0,1,1 };
		static constexpr std::array<int, 4> o3 = { 0,0,0,1 };

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> oxt{
			o3, o1, o2, o1, o3, o2, o3, o3
		};

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> oyt{
			o2, o3, o1, o2, o2, o3, o1, o1
		};

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> ozt{
			o1, o2, o3, o3, o1, o1, o2, o2
		};

		static constexpr std::array<int, 4> azyx = { 2,1,0 };
		static constexpr std::array<int, 4> azxy = { 2,0,1 };
		static constexpr std::array<int, 4> ayzx = { 1,2,0 };
		static constexpr std::array<int, 4> ayxz = { 1,0,2 };
		static constexpr std::array<int, 4> axzy = { 0,2,1 };
		static constexpr std::array<int, 4> axyz = { 0,1,2 };

		inline static constinit __declspec(align(16)) const std::array<std::array<int, 4>, 8> apit{
			azyx,
			axzy,
			ayxz,
			axyz,
			azyx,
			azxy,
			ayzx,
			ayzx
		};

	public:
		cube3d_bgra32_avx512(cube_base&& b) :
			cube3d_bgra32{ std::move(b) }
		{
			domains.resize(domains.size() + 5); // padding
		}

        constexpr static size_t simd_unit = 4;

        void apply(BGRA32* ptr, const PreCalc& pc) const {
            const auto bgra_f32 = [this](const BGRA32* ptr) {
				const auto m1 = AVX512::u8x16::load(ptr);
				const auto m2 = AVX512::convert_to<AVX512::f32x16>(m1);
				return AVX512::permute<2, 1, 0, 3>(m2);
			}(ptr);

			const auto [midi, mt] = [](const auto& bgra_f32, const auto& resolution) {
				const auto index = bgra_f32 * resolution;
				const auto floored = AVX512::floor(index);
				return std::make_tuple(AVX512::convert_to<AVX512::i32x16>(floored), index - floored);
			}(bgra_f32, pc.resolution);
            
			auto cmp_idx = [](const AVX512::f32x16& m) {
                // y z x
                const auto m1 = AVX512::permute<1, 2, 0, 0>(m);

				// x>=y y>=z z>=x
				const auto ges = AVX512::cmp_ge(m, m1).to_int();
				const auto aa = ges & 0b0111'0111'0111'0111;
				const auto ab = _pdep_u32(aa, 0b11110000'11110000'11110000'11110000); // *=16
				return AVX512::convert_to<AVX512::i32x4>(AVX512::u8x4::load(&ab));
			}(mt);

			auto myload = [](const AVX512::i32x4& adr) {
				alignas(simden::m128i) std::array<int, 4> a;
				adr.store<true>(a);
				// require align
				return AVX512::i32x16{
					AVX512::i32x4::load<true>(reinterpret_cast<const int*>(a[0])),
					AVX512::i32x4::load<true>(reinterpret_cast<const int*>(a[1])),
					AVX512::i32x4::load<true>(reinterpret_cast<const int*>(a[2])),
					AVX512::i32x4::load<true>(reinterpret_cast<const int*>(a[3])),
				};
			};

			const auto ox = myload(AVX512::set1<AVX512::i32x4>(reinterpret_cast<int>(oxt.data()->data())) + cmp_idx);
			const auto oy = myload(AVX512::set1<AVX512::i32x4>(reinterpret_cast<int>(oyt.data()->data())) + cmp_idx);
			const auto oz = myload(AVX512::set1<AVX512::i32x4>(reinterpret_cast<int>(ozt.data()->data())) + cmp_idx);
			const auto api = myload(AVX512::set1<AVX512::i32x4>(reinterpret_cast<int>(apit.data()->data())) + cmp_idx);

            const auto ap = [](const AVX512::f32x16& mt, const AVX512::i32x16& api) {
				const auto m1 = AVX512::permutev(mt, api);

				const AVX512::f32x16 m2 = [](const AVX512::f32x16& m) {
					const auto m1 = AVX512::permute<0, 0, 1, 2>(m);
					const AVX512::mask16 mask = simden::make_flag(1,
						1, 0, 0, 0,
						1, 0, 0, 0,
						1, 0, 0, 0,
						1, 0, 0, 0
					);
					return AVX512::set1<AVX512::f32x16>(m1, 1.f, mask);
				}(m1);
				
				const AVX512::f32x16 m3 = [](const AVX512::f32x16& m) {
					const AVX512::mask16 mask = simden::make_flag(1,
						0, 0, 0, 1,
						0, 0, 0, 1,
						0, 0, 0, 1,
						0, 0, 0, 1
					);
					return AVX512::set1<AVX512::f32x16>(m, 0.f, mask);
				}(m1);
				return m2 - m3;
			} (mt, api);

			auto ix = AVX512::permute<0, 0, 0, 0>(midi) + ox;
			auto iy = AVX512::permute<1, 1, 1, 1>(midi) + oy;
			auto iz = AVX512::permute<2, 2, 2, 2>(midi) + oz;
            
			auto ip = [&pc](const AVX512::i32x16& pi, const AVX512::f32x16& ap) {
				auto load_domain = [](const AVX512::i32x4& adr) {
					alignas(simden::m128i) std::array<int, 4> a;
					adr.store<true>(a);
					
					return AVX512::f32x16{
						AVX512::f32x4::load<false>(reinterpret_cast<const float*>(a[0])),
						AVX512::f32x4::load<false>(reinterpret_cast<const float*>(a[1])),
						AVX512::f32x4::load<false>(reinterpret_cast<const float*>(a[2])),
						AVX512::f32x4::load<false>(reinterpret_cast<const float*>(a[3])),
					};
				};

				static constinit const AVX512::i32x16 c12{ 12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12 };
				const auto pi12 = pi * c12;
				const auto adr = pc.data + pi12;

				static constinit const AVX512::i32x16 permutex = std::make_from_tuple<AVX512::i32x16>([] {
					std::array<int, 16> r;
					int c = 0;
					for (int i = 0; i < 4; i++) {
						for (int j = 0; j < 4; j++) {
							r[c] = j * 4 + i;
							c++;
						}
					}
					return r;
				}());

				const auto dm = AVX512::permutex(adr, permutex);
				
				const auto m1 = AVX512::fma(load_domain(AVX512::extract<AVX512::i32x4, 0>(dm)), AVX512::permute<0, 0, 0, 0>(ap), load_domain(AVX512::extract<AVX512::i32x4, 1>(dm)) * AVX512::permute<1, 1, 1, 1>(ap));
				const auto m2 = AVX512::fma(load_domain(AVX512::extract<AVX512::i32x4, 2>(dm)), AVX512::permute<2, 2, 2, 2>(ap), load_domain(AVX512::extract<AVX512::i32x4, 3>(dm)) * AVX512::permute<3, 3, 3, 3>(ap));
				const auto m3 = m1 + m2;
				return m3;
			};

			auto idx3d_m = [size = size, &pc](const AVX512::i32x16& mx, const AVX512::i32x16& my, const AVX512::i32x16& mz) {
				const auto m1 = AVX512::min_(mz, pc.sizem1) * pc.size;
				const auto m2 = m1 + AVX512::min_(my, pc.sizem1);
				const auto m3 = m2 * pc.size;
				const auto m4 = m3 + AVX512::min_(mx, pc.sizem1);
				return m4;
			};

            const auto applied = ip(idx3d_m(ix, iy, iz), ap);
			const auto result = AVX512::fma(bgra_f32, pc.afts, applied * pc.aftt);

            auto f32_to_bgra = [](BGRA32* ptr, const AVX512::f32x16& m) {
                const auto m1 = AVX512::convert_to<AVX512::i32x16>(AVX512::round(m));
				static constinit const AVX512::i32x16 c0 = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
				static constinit const AVX512::i32x16 c255 = { 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255 };
                const auto m2 = AVX512::clamp(m1, c0, c255);
				// permute and shuffle
				[](const AVX512::i32x16& x, void* ptr) {
					#ifdef SIMDEN_EMULATE_INTRINSICS
					const auto m1 = AVX512::permute<2, 1, 0, 3>(x);
					const auto m2 = AVX512::convert_to<AVX512::u8x16>(m1);
					m2.store(ptr);
					#else
					static constinit const simden::m512i shuffle_u32_to_permuted_u8{ .m512i_i8{
						8, 4, 0, 12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
						8, 4, 0, 12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
						8, 4, 0, 12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
						8, 4, 0, 12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
					} };
					const auto m1 = _mm512_shuffle_epi8(x, shuffle_u32_to_permuted_u8);
					static constinit const __mmask16 compress_mask = 0b0001'0001'0001'0001;
					_mm512_mask_compressstoreu_epi32(ptr, compress_mask, m1);
					#endif
				} (m2, ptr);
            };
            f32_to_bgra(ptr, result);
        }
	
        void apply(BGRA32* d, size_t count, float af) const {
			const auto afs = 1.f - af;
			const auto ssize = static_cast<int>(size);
			const auto ssizem1 = ssize - 1;
			const auto data = reinterpret_cast<int>(this->domains.data()->data());
			const PreCalc pc{
				.afts{
					afs, afs, afs, 1.f,
					afs, afs, afs, 1.f,
					afs, afs, afs, 1.f,
					afs, afs, afs, 1.f,
				},
				.aftt{
					af, af, af, 0.f,
					af, af, af, 0.f,
					af, af, af, 0.f,
					af, af, af, 0.f,
				},
				.resolution{
					resolution,resolution,resolution,resolution,resolution,resolution,resolution,resolution,
					resolution,resolution,resolution,resolution,resolution,resolution,resolution,resolution,
				},
				.size{
					ssize,ssize,ssize,ssize,ssize,ssize,ssize,ssize,
					ssize,ssize,ssize,ssize,ssize,ssize,ssize,ssize,
				},
				.sizem1{
					ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,
					ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,ssizem1,
				},
				.data{
					data,data,data,data,data,data,data,data,
					data,data,data,data,data,data,data,data,
				}
			};
			
			#ifndef CUBEHPP_PARALLEL_OFF
			concurrency::parallel_for(0u, count, [this, d, &pc](size_t i) {
			#else
			for (size_t i = 0; i < count; i++) {
			#endif
				apply(d + i * simd_unit, pc);
			#ifndef CUBEHPP_PARALLEL_OFF
			});
			#else
			}
			#pragma message("parallel off AVX-512")
			#endif
        }
    };

	template<class T1D, class T3D>
    class cube_bgra32_simd_tmp {
        std::variant<T1D, T3D> obj;
        using obj_t = decltype(obj);

		void save_checked(const std::filesystem::path& path, const Domain& s_domain_min, const Domain& s_domain_max) const {
			std::visit([&path, &s_domain_min, &s_domain_max](const auto& o) {
				std::ofstream ofs{ path, std::ios::binary };
				std::ostreambuf_iterator itr{ ofs };

				ofs << detail::HEAD_CUBEHPP_COMMENT;

				if (o.title.has_value()) std::format_to(itr, "{} \"{}\"\n", HEAD_TITLE, o.title.value());

				std::format_to(itr, "{} {}\n", HEAD_DOMAIN_MIN, s_domain_min);
				std::format_to(itr, "{} {}\n", HEAD_DOMAIN_MAX, s_domain_max);

				o.write_head_size(itr);

				const Domain domain_a = {
					(s_domain_max[0] - s_domain_min[0]) / 255,
					(s_domain_max[1] - s_domain_min[1]) / 255,
					(s_domain_max[2] - s_domain_min[2]) / 255,
				};
				for (size_t i : std::views::iota(size_t{ 0 }, o.get_domains_size())) {
					Domain d{
						o.domains[i][0] * domain_a[0] + s_domain_min[0],
						o.domains[i][1] * domain_a[1] + s_domain_min[1],
						o.domains[i][2] * domain_a[2] + s_domain_min[2],
					};

					std::format_to(itr, "{}\n", d);
				}
			}, obj);
		}
		
		void change_domain_range_checked(const Domain& new_domain_min, const Domain& new_domain_max) {
			std::visit([&new_domain_min, &new_domain_max](auto& o) {
				const Domain old_domain_min = o.domain_min.value_or(domain_min_default);
				const Domain old_domain_max = o.domain_max.value_or(domain_max_default);
				const Domain old_domain_dist{
					old_domain_max[0] - old_domain_min[0],
					old_domain_max[1] - old_domain_min[1],
					old_domain_max[2] - old_domain_min[2],
				};
				const Domain new_domain_dist{
					new_domain_max[0] - new_domain_min[0],
					new_domain_max[1] - new_domain_min[1],
					new_domain_max[2] - new_domain_min[2],
				};
				const Domain domain_a{
					new_domain_dist[0] / old_domain_dist[0],
					new_domain_dist[1] / old_domain_dist[1],
					new_domain_dist[2] / old_domain_dist[2],
				};

				std::for_each(
					std::execution::par_unseq,
					std::ranges::begin(o.domains), std::ranges::end(o.domains),
					[&](Domain& d) {
						d[0] = (d[0] - old_domain_min[0]) * domain_a[0] + new_domain_min[0];
						d[1] = (d[1] - old_domain_min[1]) * domain_a[1] + new_domain_min[1];
						d[2] = (d[2] - old_domain_min[2]) * domain_a[2] + new_domain_min[2];
					}
				);
				o.domain_min = new_domain_min;
				o.domain_max = new_domain_max;
			}, obj);
		}
	public:
		cube_bgra32_simd_tmp(CubeType type, cube_base&& prop) : obj{
			[type, &prop]() -> obj_t {
				switch (type) {
					case CubeType::c_1d:
						return T1D{ std::move(prop) };
					case CubeType::c_3d:
						return T3D{ std::move(prop) };
					default:
						throw type_invalid_exception{ "cube_bgra32_avx2::cube_bgra32_avx2" };
				}
			}()
		} {
			change_domain_range_checked({ 0,0,0 }, { 255,255,255 });
		}

		cube_bgra32_simd_tmp(CubeType type, std::optional<std::string>&& title, BGRA32* data) : obj{
			[type, &title, data]() mutable -> obj_t {
				cube_base prop;
				prop.title = title;
				prop.domain_min = { 0,0,0 };
				prop.domain_max = { 255,255,255 };
				switch (type) {
					case CubeType::c_1d:
						prop.size = 256;
						prop.domains.reserve(256);
						for (int x = 0; x < 256; x++) {
							prop.domains.push_back(Domain{
								static_cast<float>(data[x][2]),
								static_cast<float>(data[x + 80 * 256][1]),
								static_cast<float>(data[x + 80 * 2 * 256][0]),
							});
						}
						return T1D{ std::move(prop) };
					case CubeType::c_3d:
						prop.size = 64;
						prop.domains.reserve(64 * 64 * 64);
						for (int zy = 0; zy < 8; zy++) {
							for (int zx = 0; zx < 8; zx++) {
								for (int y = 0; y < 64; y++) {
									for (int x = 0; x < 64; x++) {
										auto& p = data[(zy * 64 + y) * 512 + zx * 64 + x];
										prop.domains.push_back(Domain{
											static_cast<float>(p[2]),
											static_cast<float>(p[1]),
											static_cast<float>(p[0]),
										});
									}
								}
							}
						}
						return T3D{ std::move(prop) };
					default:
						throw type_invalid_exception{ "cube_bgra32_avx2::cube_bgra32_avx2" };
				}
			}()
		} {}

        
		constexpr CubeType type() const {
			return std::visit([](const auto& o) { return o.type; }, obj);
		}

		constexpr size_t size() const {
			return std::visit([](const auto& o) { return o.size; }, obj);
		}

		[[nodiscard]] constexpr Domain& at(size_t i)& {
			return std::visit([i](auto& o) -> auto& { return o.domains[i]; }, obj);
		}
		[[nodiscard]] constexpr const Domain& at(size_t i) const& {
			return std::visit([i](const auto& o) -> const auto& { return o.domains[i]; }, obj);
		}

		std::optional<std::string>& title()& {
			return std::visit([](auto& o) -> auto& { return o.title; }, obj);
		}
		const std::optional<std::string>& title() const& {
			return std::visit([](auto& o) -> const auto& { return o.title; }, obj);
		}

		void apply(Domain& d) const {
			std::visit([&d](const auto& o) { d = o.index(d[0], d[1], d[2]); }, obj);
		}

		void apply(Domain& d, float af) const {
			std::visit([&d, af](const auto& o) {
                auto lerp = [](const Domain& a, const Domain& b, float t) {
                    const auto s = 1.f - t;
                    return Domain{
                        a[0] * s + b[0] * t,
                        a[1] * s + b[1] * t,
                        a[2] * s + b[2] * t,
                    };
                };
                d = lerp(d, o.index(d[0], d[1], d[2]), af);
            }, obj);
		}

		void apply(BGRA32* d, size_t count, float af) const {
			std::visit([d, count, af](const auto& o) {
                o.apply(d, (count + o.simd_unit - 1) / o.simd_unit, af);
            }, obj);
		}

		void save(const std::filesystem::path& path) const {
			std::visit([&path](const auto& o) {
				std::ofstream ofs{ path, std::ios::binary };
				std::ostreambuf_iterator itr{ ofs };

				ofs << detail::HEAD_CUBEHPP_COMMENT;

				if (o.title.has_value()) std::format_to(itr, "{} \"{}\"\n", HEAD_TITLE, o.title.value());

				if (o.domain_min.has_value()) { std::format_to(itr, "{} {}\n", HEAD_DOMAIN_MIN, o.domain_min.value()); };
				if (o.domain_max.has_value()) { std::format_to(itr, "{} {}\n", HEAD_DOMAIN_MAX, o.domain_max.value()); };

				o.write_head_size(itr);

				for (size_t i : std::views::iota(size_t{ 0 }, o.get_domains_size())) {
					std::format_to(itr, "{}\n", o.domains[i]);
				}
			}, obj);
		}

		void save(const std::filesystem::path& path, float s_domain_min, float s_domain_max) const {
			if (s_domain_min >= s_domain_max) throw invalid_domain_range_exception{};

			save_checked(path, { s_domain_min,s_domain_min,s_domain_min }, { s_domain_max,s_domain_max,s_domain_max });
		}

		void save(const std::filesystem::path& path, const Domain& s_domain_min, const Domain& s_domain_max) const {
			if (!(
				s_domain_min[0] < s_domain_max[0] &&
				s_domain_min[1] < s_domain_max[1] &&
				s_domain_min[2] < s_domain_max[2]
			)) throw invalid_domain_range_exception{};

			save_checked(path, s_domain_min, s_domain_max);
		}

		void change_domain_range(float new_domain_min, float new_domain_max) {
			if (new_domain_min >= new_domain_max) throw invalid_domain_range_exception{};

			change_domain_range_checked({ new_domain_min,new_domain_min,new_domain_min }, { new_domain_max,new_domain_max,new_domain_max });
		}

		void change_domain_range(const Domain& new_domain_min, const Domain& new_domain_max) {
			if (!(
				new_domain_min[0] < new_domain_max[0] &&
				new_domain_min[1] < new_domain_max[1] &&
				new_domain_min[2] < new_domain_max[2]
			)) throw invalid_domain_range_exception{};

			change_domain_range_checked(new_domain_min, new_domain_max);
		}
	};

	using cube_bgra32_avx2 = cube_bgra32_simd_tmp<cube1d_bgra32_avx2, cube3d_bgra32_avx2>;

	using cube_bgra32_avx512 = cube_bgra32_simd_tmp<cube1d_bgra32_avx2, cube3d_bgra32_avx512>;

	inline auto cube_bgra32_avx2_from_file(const std::filesystem::path& path) {
		cube_reader cr{ path };
		return cube_bgra32_avx2{ cr.get_type(), std::move(cr) };
	}

	inline auto cube_bgra32_avx512_from_file(const std::filesystem::path& path) {
		cube_reader cr{ path };
		return cube_bgra32_avx512{ cr.get_type(), std::move(cr) };
	}
}
