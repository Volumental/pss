/*
  Copyright (C) 2014 Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Intel Corporation nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/
#include <iterator>
#include <algorithm>
#include <oneapi/tbb/task_group.h>

#include "pss_common.h"

namespace pss {

namespace internal {

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
class merge_task {
    oneapi::tbb::task_group& tg;
    RandomAccessIterator1 xs, xe;
    RandomAccessIterator2 ys, ye;
    RandomAccessIterator3 zs;
    Compare comp;
    bool destroy;
public:
    merge_task( oneapi::tbb::task_group& tg_, RandomAccessIterator1 xs_, RandomAccessIterator1 xe_, RandomAccessIterator2 ys_, RandomAccessIterator2 ye_, RandomAccessIterator3 zs_, bool destroy_, Compare comp_ ) :
        tg(tg_), xs(xs_), xe(xe_), ys(ys_), ye(ye_), zs(zs_), comp(comp_), destroy(destroy_)
    {}
    void operator()() const;
};

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
void merge_task<RandomAccessIterator1,RandomAccessIterator2,RandomAccessIterator3,Compare>::operator ()() const {
    const size_t MERGE_CUT_OFF = 2000;
    auto n = (xe-xs) + (ye-ys);
    if( (size_t) n <= MERGE_CUT_OFF ) {
        serial_move_merge( xs, xe, ys, ye, zs, comp );
        if( destroy ) {
            serial_destroy(xs,xe);
            serial_destroy(ys,ye);
        }
        return;
    } else {
        RandomAccessIterator1 xm;
        RandomAccessIterator2 ym;
        if( xe-xs < ye-ys  ) {
            ym = ys+(ye-ys)/2;
            xm = std::upper_bound(xs,xe,*ym,comp);
        } else {
            xm = xs+(xe-xs)/2;
            ym = std::lower_bound(ys,ye,*xm,comp);
        }
        RandomAccessIterator3 zm = zs + ((xm-xs) + (ym-ys));
        //tbb::task* right = new( allocate_additional_child_of(*parent()) ) merge_task( xm, xe, ym, ye, zm, destroy, comp );
        //spawn(*right);
        //recycle_as_continuation();
        tg.run(merge_task( tg, xm, xe, ym, ye, zm, destroy, comp ));  // right
        tg.run(merge_task( tg, xm, xm, ym, ym, zm, destroy, comp ));  // left
        //xe = xm;
        //ye = ym;
    }
}

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
class stable_sort_task {
    oneapi::tbb::task_group& tg;
    RandomAccessIterator1 xs, xe;
    RandomAccessIterator2 zs;
    Compare comp;
    signed char inplace;
public:
    stable_sort_task(oneapi::tbb::task_group& tg_, RandomAccessIterator1 xs_, RandomAccessIterator1 xe_, RandomAccessIterator2 zs_, int inplace_, Compare comp_ ) :
        tg(tg_), xs(xs_), xe(xe_), zs(zs_), comp(comp_), inplace(inplace_)
    {}
    void operator()() const;
};

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void stable_sort_task<RandomAccessIterator1, RandomAccessIterator2, Compare>::operator()() const {
    const size_t SORT_CUT_OFF = 500;
    if ((size_t) (xe - xs) <= SORT_CUT_OFF) {
        stable_sort_base_case(xs, xe, zs, inplace, comp);
        return;
    } else {
        RandomAccessIterator1 xm = xs + (xe - xs) / 2;
        RandomAccessIterator2 zm = zs + (xm - xs);
        RandomAccessIterator2 ze = zs + (xe - xs);
        oneapi::tbb::parallel_invoke(
            stable_sort_task(tg, xm,xe,zm,!inplace, comp), // right
            stable_sort_task(tg, xs,xm,zs,!inplace, comp)); // left

        if (inplace)
            merge_task<RandomAccessIterator2,RandomAccessIterator2,RandomAccessIterator1,Compare>(tg, zs, zm, zm, ze, xs, inplace==2, comp)();
        else
            merge_task<RandomAccessIterator1,RandomAccessIterator1,RandomAccessIterator2,Compare>(tg, xs, xm, xm, xe, zs, false, comp)();
    }
}

} // namespace internal

template<typename RandomAccessIterator, typename Compare>
void parallel_stable_sort( RandomAccessIterator xs, RandomAccessIterator xe, Compare comp ) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
    if( internal::raw_buffer z = internal::raw_buffer( sizeof(T)*(xe-xs) ) ) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        internal::raw_buffer buf( sizeof(T)*(xe-xs) );
        oneapi::tbb::task_group tg;
        tg.run(internal::stable_sort_task<RandomAccessIterator,T*,Compare>( tg, xs, xe, (T*)buf.get(), 2, comp ));
        tg.wait();
        // task::spawn_root_and_wait(*new( task::allocate_root() ) internal::stable_sort_task<RandomAccessIterator,T*,Compare>( xs, xe, (T*)buf.get(), 2, comp ));
    } else
        // Not enough memory available - fall back on serial sort
        std::stable_sort( xs, xe, comp );
}

} // namespace pss
